

import torch


class yololoss(torch.nn.Module):
    def __init__(self):
        super(yololoss, self).__init__()

    def iou(self, box1, box2):
        # print("box1,box2:",box1,box2)
        N = box1.size()[0]
        M = box2.size()[0]
        # n,x,y,w,h
        x1y1 = torch.autograd.Variable(torch.zeros(box1.size()))
        x1y1[:, 0] = box1[:, 0] - box1[:, 2] * 1 / 2.  # x1=x-1/2*w
        x1y1[:, 2] = box1[:, 0] + box1[:, 2] * 1 / 2.  # x2=x+1/2*w
        x1y1[:, 1] = box1[:, 1] - box1[:, 3] * 1 / 2.  # y1=y-1/2*h
        x1y1[:, 3] = box1[:, 1] - box1[:, 3] * 1 / 2.  # y2=y+1/2*h
        x2y2 = torch.autograd.Variable(torch.zeros(box2.size()))
        x2y2[:, 0] = box2[:, 0] - box2[:, 2] * 1 / 2.  # x1=x-1/2*w
        x2y2[:, 2] = box2[:, 0] + box2[:, 2] * 1 / 2.  # x2=x+1/2*w
        x2y2[:, 1] = box2[:, 1] - box2[:, 3] * 1 / 2.  # y1=y-1/2*h
        x2y2[:, 3] = box2[:, 1] - box2[:, 3] * 1 / 2.  # y2=y+1/2*h
        # print("x1y1:",x1y1.size(),x2y2.size())
        # print(x1y1[:,:2].unsqueeze(1).expand(N,M,2).size())
        # print(x2y2[:,2:4].unsqueeze(0).expand(N,M,2).size())
        lt = torch.max(x1y1[:, :2].unsqueeze(1).expand(N, M, 2), x2y2[:, :2].unsqueeze(0).expand(N, M, 2))  # x1 y1
        rb = torch.min(x1y1[:, 2:4].unsqueeze(1).expand(N, M, 2), x2y2[:, 2:4].unsqueeze(0).expand(N, M, 2))  # x2,y2
        # print("x1y1,x2y2:",x1y1,x2y2)
        wh = rb - lt
        # wh=(wh>0).float()
        # print("wh:{}".format(wh))
        # wh[wh_mask.float()] = 1
        # print("wh size:{},wh:{}".format(wh.size(),wh))
        inter = wh[:, :, 0] * wh[:, :, 1]
        # print("inter:{}".format(inter.data))
        wh1 = x1y1[:, 2:4] - x1y1[:, :2]
        area1 = wh1[:, 0] * wh1[:, 1] + 100
        # print("area1:{}".format(area1.data))
        wh2 = x2y2[:, 2:4] - x2y2[:, :2]
        area2 = wh2[:, 0] * wh2[:, 1] + 100
        # print("area2:{}".format(area2.data))
        iou = inter / (area2 + area1 - inter)
        # print("iou:{}".format(iou.data))
        return iou

    def forward(self, pred, target):
        with torch.cuda.device(0):
            N = target.size()[0]
            # print(N)
            # print("pred",pred.size())
            # print("target",target.size())
            co_mask = target[:, :, :, 4] > 0  # cell have object
            no_mask = target[:, :, :, 4] == 0  # cell have no object
            # print("co_mask",co_mask.size(),"no_mask",no_mask.size())

            # target and pred contain object
            co_target = target[co_mask.unsqueeze(3).expand_as(target)].view(-1, 30)
            # print(co_mask.unsqueeze(3).expand_as(target).size())
            co_pred = pred[co_mask.unsqueeze(3).expand_as(target)].view(-1, 30)

            # target and pred do not contain object
            no_target = target[no_mask.unsqueeze(3).expand_as(target)].view(-1, 30)
            no_pred = pred[no_mask.unsqueeze(3).expand_as(target)].view(-1, 30)
            # print("co_target:{},no_target:{}".format(co_target.size(),no_target.size()))
            # print("co_pred:{},no_pred:{}".format(co_pred.size(),no_pred.size()))

            # 1.loss:if cell have no object only compute confidence loss
            temp = torch.cuda.ByteTensor(no_target.size())
            temp.zero_()
            # print('temp',temp.size())
            temp[:, 4] = 1;
            temp[:, 9] = 1
            no_pred_temp = no_pred[temp].view(-1, 2)
            no_target_temp = no_target[temp].view(-1, 2)
            # print("no_target_temp:{},no_pred_temp:{}".format(no_target_temp.view(1,-1),no_pred_temp.size()))
            loss_no = 0.5 * torch.nn.functional.mse_loss(no_pred_temp, no_target_temp, size_average=False)
            # print("loss_no:{}".format(loss_no.data))

            # 2 class loss: cell have object
            co_target_class = co_target[:, 10:].contiguous().view(-1, 20)
            co_pred_class = co_pred[:, 10:].contiguous().view(-1, 20)
            loss_class = torch.nn.functional.mse_loss(co_target_class, co_pred_class, size_average=False)

            # 3 contain object and reponsible for bound box predict
            #
            co_co_target = co_target[:, :10].contiguous().view(-1, 5)
            co_co_pred = co_pred[:, :10].contiguous().view(-1, 5)
            co_re_mask = torch.cuda.ByteTensor(co_co_target.size()[0]).zero_()
            co_no_mask = torch.cuda.ByteTensor(co_co_target.size()[0]).zero_()
            # print("co_co_target:{},co_co_pred:{},loss_class:{}".format(co_co_target.size(),co_co_pred.size(),loss_class.data[0]))
            # print('co_re_mask',co_re_mask.size())
            # print('co_no_mask',co_no_mask.size())
            # cell have object
            for i in range(0, co_co_target.size()[0], 2):
                target_box = co_co_target[i, :].view(1, -1)
                pred_box = co_co_pred[i:i + 2, :]
                # print("pred_box:".format(pred_box))
                # print("1",target_box.size(),pred_box.size())
                iou = self.iou(pred_box, target_box)
                max_iou, max_index = iou.max(0)
                # print("max_index:{},max_iou:{}".format(max_index,max_iou))
                co_re_mask[i + max_index.data[0]] = 1
                co_no_mask[i + 1 - max_index.data[0]] = 1

            # have obj loss
            temp_target = co_co_target[co_re_mask.unsqueeze(1).expand_as(co_co_target)].view(-1, 5)
            temp_pred = co_co_pred[co_re_mask.unsqueeze(1).expand_as(co_co_target)].view(-1, 5)
            # print('temp',temp.size())
            # print('temp_target',temp_target.size())
            # print('temp_pred',temp_pred.size())
            # print(temp_pred[:,2][0])
            loss_x = 5. * torch.nn.functional.mse_loss(temp_target[:, 0], temp_pred[:, 0], size_average=False)
            loss_y = 5. * torch.nn.functional.mse_loss(temp_target[:, 1], temp_pred[:, 1], size_average=False)
            loss_w = 5. * torch.nn.functional.mse_loss(torch.sqrt(temp_target[:, 2]), torch.sqrt(temp_pred[:, 2]),
                                                       size_average=False)
            loss_h = 5. * torch.nn.functional.mse_loss(torch.sqrt(temp_target[:, 3]), torch.sqrt(temp_pred[:, 3]),
                                                       size_average=False)
            loss_c = torch.nn.functional.mse_loss(temp_target[:, 4], temp_pred[:, 4], size_average=False)

            # not reponseble for obj
            temp_target = co_co_target[co_no_mask.unsqueeze(1).expand_as(co_co_target)].view(-1, 5)
            temp_pred = co_co_pred[co_no_mask.unsqueeze(1).expand_as(co_co_target)].view(-1, 5)
            loss_nore = 0.5 * torch.nn.functional.mse_loss(temp_target[:, 4], temp_pred[:, 4], size_average=False)
            loss = loss_no + loss_class + loss_x + loss_y + loss_w + loss_h + loss_c + loss_nore
        # print(loss_no.data[0],loss_class.data[0],loss_x.data[0],loss_y.data[0],loss_w.data[0],loss_h.data[0],loss_c.data[0],loss_nore.data[0])
        return loss / N


if __name__ == "__main__":
    # from yoloLoss import yoloLoss

    # loss1=yololoss()
    loss2 = yololoss()
    pred = torch.autograd.Variable(torch.Tensor(1, 7, 7, 30).uniform_(0, 1))
    target = torch.autograd.Variable(torch.randn(1, 7, 7, 30).uniform_(0, 1))
    # l1=loss1(pred,target).data[0]
    # print(l1)
    l2 = loss2(pred, target).item()
    print(l2)
