from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
from tqdm import tqdm, trange
from utils.metric import accuracy, AverageMeter, Timer

class SI(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, agent_config,model):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super(SI, self).__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.model = model
        self.criterion_fn = nn.CrossEntropyLoss()
        # if agent_config['gpuid'][0] >= 0:
        #     self.cuda()
        #     self.gpu = True
        # else:
        #     self.gpu = False
        self.gpu=False
        self.init_optimizer()
        self.reset_optimizer = False
        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
                                    # Set a interger here for the incremental class scenario
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.regularization_terms = {}
        self.task_count = 0
        # self.online_reg = True  # True: There will be only one importance matrix and previous model parameters
        # # False: Each task has its own importance matrix and model parameters

        self.online_reg = True  # Original SI works in an online updating fashion
        self.damping_factor = 0.1
        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=0.1)


    def criterion(self, inputs, targets, tasks, regularization=True, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
        preds=inputs
        pred = preds
        # if isinstance(self.valid_out_dim, int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
        #     pred = preds['All'][:,:self.valid_out_dim]
        loss = self.criterion_fn(pred, targets)

        ###ana hai

        #TODO
        #change done between default and L2 methods
        if regularization and len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            for i, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            loss += self.config['reg_coef'] * reg_loss
        return loss

    # def update_model(self, inputs, targets, tasks):
    #     #     out = self.forward(inputs)
    #     #     loss = self.criterion(out, targets, tasks)
    #     #     self.optimizer.zero_grad()
    #     #     loss.backward()
    #     #     self.optimizer.step()
    #     #     return loss.detach(), out

    def forward(self, x):
        return self.model.forward(x)

    def update_model(self, inputs, targets, tasks):

        unreg_gradients = {}

        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2. Collect the gradients without regularization term
        out = self.model(inputs)
        loss = out[0]
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        # 3. Normal update with regularization
        loss = self.criterion(out, targets, tasks, regularization=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0

        return loss.detach(), out

    def learn_batch(self, train_loader, val_loader=None):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        self.model.zero_grad()
        # epoch_iterator = tqdm(train_loader, desc="Iteration", disable=False)
        # global_step = 0
        # epochs_trained = 0
        # steps_trained_in_current_epoch = 0
        # train_iterator = trange(
        #     epochs_trained, int(self.config['schedule'][-1]), desc="Epoch", disable=False,
        # )

        # for _ in train_iterator:
        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            self.scheduler.step(epoch)
            for param_group in self.optimizer.param_groups:
                self.log('LR:',param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            for i, (inputs_1,inputs_2,inputs_3,target) in enumerate(train_loader):
                #changed here for creating 2d tensor
                input = torch.stack([inputs_1,inputs_2,inputs_3]).reshape((8,-1))
                task='mrpc'

                data_time.update(data_timer.toc())  # measure data loading time

                if self.gpu:
                    input = input.cuda()
                    target = target.cuda()

                loss, output = self.update_model(input, target, task)
                input = input.detach()
                target = target.detach()

                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, input.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

                if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                    self.log('[{0}/{1}]\t'
                          '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                          '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))

            self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

            # Evaluate the performance of current task
            if val_loader != None:
                self.validation(val_loader)

        #from regularization

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance(train_loader)

        # Save the weight and importance of weights of current task
        self.task_count += 1
        if self.online_reg and len(self.regularization_terms)>0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {'importance':importance, 'task_param':task_param}
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}


    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            output = self.predict(input)

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, target, task, acc)

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc.avg

    def calculate_importance(self, dataloader):
        self.log('Computing SI')
        assert self.online_reg,'SI needs online_reg=True'

        # Initialize the importance matrix
        if len(self.regularization_terms)>0: # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # It is in the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
            prev_params = self.initial_params

        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n]/(delta_theta**2 + self.damping_factor)
            self.w[n].zero_()

        return importance


    def learn_stream(self, data, label):
        assert False,'No implementation yet'

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model,torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

def accumulate_acc(output, target, task, meter):
    if 'All' in output.keys(): # Single-headed model
        meter.update(accuracy(output['All'], target), len(target))
    else:  # outputs from multi-headed (multi-task) model
        for t, t_out in output.items():
            inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
            if len(inds) > 0:
                t_out = t_out[inds]
                t_target = target[inds]
                meter.update(accuracy(t_out, t_target), len(inds))

    return meter
