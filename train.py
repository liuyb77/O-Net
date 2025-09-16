# torch and visulization
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet,ONet3gaigai,ONet3gaigaifr,ONet3gaigaifrgai,ONet3gaigaifrgai2,ONet3gaigaifrgai3,ONet3gaigaifrgai3trans,DNANet2
from model.improve import ResAT,ResAT2
from model.improvezj import ResATZJ
class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = ONet3gaigaifrgai3(num_classes=1,input_channels=args.in_channels, block=ResAT, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model


        from torchsummary import summary
        summary(model, input_size=(args.in_channels, args.crop_size, args.crop_size))

        def print_model_summary(model):
            total_params = 0
            for name, param in model.named_parameters():
                param_count = param.numel()
                total_params += param_count
                #print(f"{name}: {param.size()} | Parameters: {param_count}")

            print(f"Total Parameters: {total_params}")

        # 在模型初始化后调用该函数
        print_model_summary(model)

        self.model      = model

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()

        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()
            if args.deep_supervision == 'True':
                preds= self.model(data)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, labels)
                loss /= len(preds)
            else:
               pred = self.model(data)
               loss = SoftIoULoss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC .update(pred, labels)
                self.mIoU.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU ))
            test_loss=losses.avg
        # save high-performance model
        save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())
        #self.best_iou=mean_IOU
        if mean_IOU > self.best_iou:
            self.best_iou=mean_IOU
        #print('4', self.best_iou)

def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)





