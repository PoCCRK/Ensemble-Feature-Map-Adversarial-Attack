import torch
from torchattacks.attack import Attack

class MBEL2(Attack):
    r"""
    Ensemble-based approach in the paper
    'Delving into Transferable Adversarial Examples and Black-box Attacks'
    [https://arxiv.org/abs/1611.02770v3]

    Distance Measure : L2

    The method that transfers better by avoiding dependence on any specific model.
    It uses k models with softmax outputs, notated as J1, ..., Jk, and solves:
    argmin_{ -log ( sum_{i=1}^{k} alpha_i * J_i(x') * 1_y )  + eta * d(x,x') }
    where d(x,x') = sqrt ( sum_i{x_i' - x_i}^2 / N ), root mean square deviation, i.e., RMSD

    Arguments:
        models ([nn.Module]): a list of ensemble models to attack.
        eta (float): parameter for box-constraint. (Default: 0)
        alpha ([float]): parameter for weights of models. (Default: None)
        steps (int): number of steps. (Default: 100)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default eta=0, the L2 is not limited.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
                        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = MBEL2(models, eta=1e-2, steps=100, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: sum(alpha) == 1 is not checked. default eta is 0.

    """

    def __init__(self, models, eta=0.0, eps=None, steps=100, lr=2/255, alpha=None):
        if not models:
            raise ValueError("No model in models")
        super().__init__("MBEL2", model=models[0])
        self.num_models = len(models)
        self.models = models
        self._supported_mode = ['default', 'targeted']
        self.eta = eta
        self.eps = eps
        if alpha:
            if len(alpha) == self.num_models:
                # sum(alpha) == 1 is not checked
                self.alpha = alpha
            else:
                raise ValueError("number of models and number of alpha not match")
        else:
            a = 1/self.num_models
            self.alpha = [a for i in range(self.num_models)]
        self.steps = steps
        self.lr = lr

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        adv_images = images.clone().detach()

        n_classes = 0

        MSELoss = torch.nn.MSELoss(reduction='none')
        Flatten = torch.nn.Flatten()
        Softmax = torch.nn.Softmax(dim=1)

        optimizer = torch.optim.Adam([adv_images], lr=self.lr)

        # check the n_classes and one_hot of labels, take n_classes as a parameter if you want
        with torch.no_grad():
            outputs = self.model(images)
            #print(outputs.shape, outputs)
            n_classes = len(outputs[0])
            if self._targeted:
                target_one_hot_labels = torch.eye(n_classes)[target_labels].to(self.device)
            else:
                one_hot_labels = torch.eye(n_classes)[labels].to(self.device)

        for step in range(self.steps):
            adv_images.requires_grad = True

            # L2 loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            
            # loss
            for i, model in enumerate(self.models):
                outputs = model(adv_images)
                #print(outputs.sum().item(), outputs)
                outputs = Softmax(outputs)
                ensemble_outputs = outputs*self.alpha[i] if i==0 else ensemble_outputs + outputs*self.alpha[i]
            #print(ensemble_outputs.sum().item(), ensemble_outputs)  
            
            if self._targeted:
                selected = torch.masked_select(ensemble_outputs, target_one_hot_labels.bool())
                loss = - torch.log(selected).sum()
            else:
                selected = torch.masked_select(ensemble_outputs, one_hot_labels.bool())
                loss = - torch.log(1 - selected).sum()

            # cost to minimize
            cost = loss + self.eta*L2_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            if self.eps:
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    

class MBELINF(Attack):
    r"""
    Ensemble-based approach in the paper, we make a Linf version.
    'Delving into Transferable Adversarial Examples and Black-box Attacks'
    [https://arxiv.org/abs/1611.02770v3]

    Distance Measure : Linf

    The method that transfers better by avoiding dependence on any specific model.
    It uses k models with softmax outputs, notated as J1, ..., Jk, and solves:
    argmin_{|x-x'|<eps} { -log ( sum_{i=1}^{k} {alpha_i * J_i(x') * 1_y )} }

    Arguments:
        models ([nn.Module]): a list of ensemble models to attack.
        rho (float): step size of grad sign. (Default: 1/255)
        eps (float): maximum perturbation. (Default: 0.1)
        alpha ([float]): parameter for weights of models. (Default: None)
        steps (int): number of steps. (Default: 50)

    .. warning:: use eta and eps as BIM to update adv_images.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
                        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = MBELINF(models, eta=1/255, eps=0.1, steps=50)
        >>> adv_images = attack(images, labels)

    .. note:: sum(alpha) == 1 is not checked.

    """

    def __init__(self, models, rho=1/255, eps=0.1, steps=50, alpha=None):
        if not models:
            raise ValueError("No model in models")
        super().__init__("MBELINF", model=models[0])
        self.num_models = len(models)
        self.models = models
        self._supported_mode = ['default', 'targeted']
        self.rho = rho
        self.eps = eps
        if alpha:
            if len(alpha) == self.num_models:
                # sum(alpha) == 1 is not checked
                self.alpha = alpha
            else:
                raise ValueError("number of models and number of alpha not match")
        else:
            a = 1/self.num_models
            self.alpha = [a for i in range(self.num_models)]
        self.steps = steps

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        adv_images = images.clone().detach()
        n_classes = 0
        Softmax = torch.nn.Softmax(dim=1)
        
        # check the n_classes and one_hot of labels, take n_classes as a parameter if you want
        with torch.no_grad():
            outputs = self.model(images)
            n_classes = len(outputs[0])
            if self._targeted:
                target_one_hot_labels = torch.eye(n_classes)[target_labels].to(self.device)
            else:
                one_hot_labels = torch.eye(n_classes)[labels].to(self.device)

        for step in range(self.steps):
            adv_images.requires_grad = True

            # loss
            for i, model in enumerate(self.models):
                outputs = model(adv_images)
                outputs = Softmax(outputs)
                ensemble_outputs = outputs*self.alpha[i] if i==0 else ensemble_outputs + outputs*self.alpha[i]
            #print(ensemble_outputs.sum().item(), ensemble_outputs)  
            
            if self._targeted:
                selected = torch.masked_select(ensemble_outputs, target_one_hot_labels.bool())
                loss = - torch.log(selected).sum()
            else:
                selected = torch.masked_select(ensemble_outputs, one_hot_labels.bool())
                loss = - torch.log(1 - selected).sum()

            # cost to minimize
            cost = loss

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.rho*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images