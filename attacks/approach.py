import torch, torchvision
from torchattacks.attack import Attack

class APPROACH(Attack):
    r"""
    Our Approach
    argmin_{|x-x'|<eps}
           { - log ( sum_{i=1}^{k} {alpha_i * J_i(x') * 1_y}s )
             + eta * d(F(x), F(x')) }

    Distance Measure : LINF

    Arguments:
        models ([nn.Module]): a list of ensemble models to attack.
        eta (float): parameter for feature loss. (Default: 0.5)
        rho (float): maximum perturbation. (Default: 1/255)
        eps (float): maximum perturbation. (Default: 0.1)
        alpha ([float]): parameter for weights of models. (Default: None)
        steps (int): number of steps. (Default: 50)

    .. warning:: Under develop.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
                        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = APPROACH(models, feature, eta=0.5, rho=2/255, eps=0.3, steps=50)
        >>> adv_images = attack(images, labels)

    .. note:: sum(alpha) == 1 is not checked.
    """
    
    def __init__(self, models, eta=0.1, rho=1/255, eps=8/255,
                 steps=50, alpha=None, target_images_list=None):
        if not models:
            raise ValueError("No model in models")
        super().__init__("APPROACH2", model=models[0])
        self.num_models = len(models)
        self.models = models
        self._supported_mode = ['default', 'targeted']
        self.eta = eta
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
        self.target_image = target_images_list

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        Softmax = torch.nn.Softmax(dim=1)
        MSELoss = torch.nn.MSELoss(reduction='none')
        Flatten = torch.nn.Flatten()
        criterion = torch.nn.CrossEntropyLoss()
        blurrer = torchvision.transforms.GaussianBlur(kernel_size=(5, 5))

        adv_images = images.clone().detach()
        # Starting at a uniformly random point
        #adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        #adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        # check the n_classes and one_hot of labels, take n_classes as a parameter if you want
        with torch.no_grad():
            outputs = self.model(images)
            n_classes = len(outputs[0])
            if self._targeted:
                target_one_hot_labels = torch.eye(n_classes)[target_labels].to(self.device)
            else:
                one_hot_labels = torch.eye(n_classes)[labels].to(self.device)

            origin_features = [ model.features(images) for model in self.models ]

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
            #loss = self.f(ensemble_outputs, labels).sum()

            # feature loss
            if self._targeted:
                target_image = [self.target_image[l] for l in target_labels]
                target_image = torch.stack(target_image, dim=0).to(self.device)
                #print(target_image.shape)

                for i, model in enumerate(self.models):
                    adv_features = model.features(adv_images)
                    features_dim = adv_features.shape[1]
                    target_features = model.features(target_image)
                    
                    feature_distance = MSELoss(Flatten(target_features),
                                               Flatten(adv_features)).sum(dim=1)
                    feature_distance = feature_distance / features_dim
                    #print(features_dim, feature_distance)
                    ensemble_features = feature_distance*self.alpha[i] if i==0 else ensemble_features + feature_distance*self.alpha[i]
                feature_loss = ensemble_features.sum()
            else:
                for i, model in enumerate(self.models):
                    adv_features = model.features(adv_images)
                    features_dim = adv_features.shape[1]
                    feature_distance = MSELoss(Flatten(origin_features[i]),
                                               Flatten(adv_features)).sum(dim=1)
                    feature_distance = feature_distance / features_dim
                    ensemble_features = feature_distance*self.alpha[i] if i==0 else ensemble_features + feature_distance*self.alpha[i]
                feature_loss = - ensemble_features.sum()

            # cost to minimize
            #print(loss, feature_loss)
            cost = loss + self.eta * feature_loss

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.rho*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            #delta = blurrer(delta)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class APPROACH2(Attack):
    r"""
    Our Approach
    argmin_{|x-x'|<eps}
           { - log ( sum_{i=1}^{k} {alpha_i * J_i(x') * 1_y}s )
             + eta1 * d(F(x), F(x') + eta2 * d(x, x')) }

    Distance Measure : L2

    Arguments:
        models ([nn.Module]): a list of ensemble models to attack.
        eta1 (float): parameter for feature loss. (Default: 0.5)
        eta2 (float): parameter for L2 loss. (Default: 0.0)
        lr (float): maximum perturbation. (Default: 1/255)
        eps (float): maximum perturbation. (Default: 8/255)
        alpha ([float]): parameter for weights of models. (Default: None)
        steps (int): number of steps. (Default: 50)

    .. warning:: Under develop.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,
                        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = APPROACH(models, feature, eta=0.5, rho=2/255, eps=0.3, steps=50)
        >>> adv_images = attack(images, labels)

    .. note:: sum(alpha) == 1 is not checked.
    """
    
    def __init__(self, models, eta1=1.0, eta2=0.0, lr=1/255, eps=8/255,
                 steps=50, alpha=None, target_images_list=None):
        if not models:
            raise ValueError("No model in models")
        super().__init__("APPROACH2", model=models[0])
        self.num_models = len(models)
        self.models = models
        self._supported_mode = ['default', 'targeted']
        self.eta1 = eta1
        self.eta2 = eta2
        self.lr = lr
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
        self.target_image = target_images_list

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        Softmax = torch.nn.Softmax(dim=1)
        MSELoss = torch.nn.MSELoss(reduction='none')
        Flatten = torch.nn.Flatten()
        criterion = torch.nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        # Starting at a uniformly random point
        #adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        #adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        optimizer = torch.optim.Adam([adv_images], lr=self.lr)
        
        # check the n_classes and one_hot of labels, take n_classes as a parameter if you want
        with torch.no_grad():
            outputs = self.model(images)
            n_classes = len(outputs[0])
            if self._targeted:
                target_one_hot_labels = torch.eye(n_classes)[target_labels].to(self.device)
            else:
                one_hot_labels = torch.eye(n_classes)[labels].to(self.device)

            origin_features = [ model.features(images) for model in self.models ]

        for step in range(self.steps):
            adv_images.requires_grad = True

            # L2 loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            
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
            #loss = self.f(ensemble_outputs, labels).sum()

            # feature loss
            if self._targeted:
                target_image = [self.target_image[l] for l in target_labels]
                target_image = torch.stack(target_image, dim=0).to(self.device)
                #print(target_image.shape)

                for i, model in enumerate(self.models):
                    adv_features = model.features(adv_images)
                    features_dim = adv_features.shape[1]
                    target_features = model.features(target_image)
                    
                    feature_distance = MSELoss(Flatten(target_features),
                                               Flatten(adv_features)).sum(dim=1)
                    feature_distance = feature_distance / features_dim
                    #print(features_dim, feature_distance)
                    ensemble_features = feature_distance*self.alpha[i] if i==0 else ensemble_features + feature_distance*self.alpha[i]
                feature_loss = ensemble_features.sum()
            else:
                for i, model in enumerate(self.models):
                    adv_features = model.features(adv_images)
                    features_dim = adv_features.shape[1]
                    feature_distance = MSELoss(Flatten(origin_features[i]),
                                               Flatten(adv_features)).sum(dim=1)
                    feature_distance = feature_distance / features_dim
                    ensemble_features = feature_distance*self.alpha[i] if i==0 else ensemble_features + feature_distance*self.alpha[i]
                feature_loss = - ensemble_features.sum()

            # cost to minimize
            #print(loss, feature_loss)
            cost = loss + self.eta1 * feature_loss + self.eta2 * L2_loss

            # Update adversarial images
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images