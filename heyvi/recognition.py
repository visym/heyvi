import os
import sys
import random
import torch
import vipy
import vipy.data.meva
import shutil
import numpy as np
from vipy.util import remkdir, filetail, readlist, tolist, filepath
from datetime import datetime
from heyvi.model.yolov3.network import Darknet
import vipy.activity
import itertools
import copy
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torch.utils.data
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl
import json
import math
import heyvi.label
import heyvi.model.ResNets_3D_PyTorch.resnet

try:
    import scipy.special
except:
    pass 


class ActivityRecognition(object):
    def __init__(self, pretrained=True):
        self.net =  None
        self._class_to_index = {}
        self._index_to_class = None
        self._num_frames = 0

    def class_to_index(self, c=None):
        return self._class_to_index if c is None else self._class_to_index[c]
    
    def index_to_class(self, index=None):
        if self._index_to_class is None:
            self._index_to_class = {v:k for (k,v) in self.class_to_index().items()}  # cache
        return self._index_to_class if index is None else self._index_to_class[index]
    
    def classlist(self):
        return [k for (k,v) in sorted(list(self.class_to_index().items()), key=lambda x: x[0])]  # sorted in index order

    def num_classes(self):
        return len(set(self.index_to_class().keys()))

    def fromindex(self, k):
        index_to_class = self.index_to_class()
        assert k in index_to_class, "Invalid class index '%s'" % (str(k))
        return index_to_class[k]

    def label_confidence(self, video=None, tensor=None, threshold=None):
        raise
        logits = self.__call__(video, tensor)
        conf = [[(self.index_to_class(j), s[j]) for j in i[::-1] if threshold is None or s[j]>threshold] for (s,i) in zip(logits, np.argsort(logits, axis=1))]
        return conf if len(logits) > 1 else conf[0]

    def activity(self, video, threshold=None):
        (c,s) = zip(*self.label_confidence(video=video, threshold=None))
        return vipy.activity.Activity(startframe=0, endframe=self._num_frames, category=c[0], actorid=video.actorid(), confidence=s[0]) if (threshold is None or s[0]>threshold) else None
            
    def top1(self, video=None, tensor=None, threshold=None):
        raise
        return self.topk(k=1, video=video, tensor=tensor, threshold=threshold)

    def topk(self, k, video=None, tensor=None, threshold=None):
        raise
        logits = self.__call__(video, tensor)
        topk = [[self.index_to_class(j) for j in i[-k:][::-1] if threshold is None or s[j] >= threshold] for (s,i) in zip(logits, np.argsort(logits, axis=1))]
        return topk if len(topk) > 1 else topk[0]

    def temporal_support(self):
        return self._num_frames

    def totensor(self, training=False):
        raise

    def binary_vector(self, categories):
        y = np.zeros(len(self.classlist())).astype(np.float32)
        for c in tolist(categories):
            y[self.class_to_index(c)] = 1
        return torch.from_numpy(y).type(torch.FloatTensor)
        
    
    
class PIP_250k(pl.LightningModule, ActivityRecognition):
    """Activity recognition using people in public - 250k stabilized"""
    
    def __init__(self, pretrained=True, deterministic=False, modelfile=None, mlbl=False, mlfl=True, unitnorm=False):

        # FIXME: remove dependencies here
        from heyvi.model.pyvideoresearch.bases.resnet50_3d import ResNet503D, ResNet3D, Bottleneck3D
        import heyvi.model.ResNets_3D_PyTorch.resnet
        
        super().__init__()
        self._input_size = 112
        self._num_frames = 16        
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._mlfl = mlfl
        self._mlbl = mlbl
        self._unitnorm = unitnorm

        if deterministic:
            np.random.seed(42)

        self._class_to_weight = {'car_drops_off_person': 1.4162811344926518, 'car_picks_up_person': 1.4103618337303332, 'car_reverses': 1.0847976470131024, 'car_starts': 1.0145749063037774, 'car_stops': 0.6659236295324015, 'car_turns_left': 2.942269221156227, 'car_turns_right': 1.1077783089040996, 'hand_interacts_with_person_highfive': 2.793646013249904, 'person': 0.4492053391155403, 'person_abandons_object': 1.0944029463871692, 'person_carries_heavy_object': 0.5848339202761978, 'person_closes_car_door': 0.8616907697519004, 'person_closes_car_trunk': 1.468393359799126, 'person_closes_facility_door': 0.8927495923340439, 'person_embraces_person': 0.6072654081071569, 'person_enters_car': 1.3259274145537951, 'person_enters_scene_through_structure': 0.6928103470838287, 'person_exits_car': 1.6366577285051707, 'person_exits_scene_through_structure': 0.8368692178634396, 'person_holds_hand': 1.2378881634203558, 'person_interacts_with_laptop': 1.6276031281396193, 'person_loads_car': 2.170167410167583, 'person_opens_car_door': 0.7601817241565009, 'person_opens_car_trunk': 1.7255285914206204, 'person_opens_facility_door': 0.9167411017455822, 'person_picks_up_object_from_floor': 1.123251610875369, 'person_picks_up_object_from_table': 3.5979689180114205, 'person_purchases_from_cashier': 7.144918373837205, 'person_purchases_from_machine': 5.920886403645001, 'person_puts_down_object_on_floor': 0.7295795950752353, 'person_puts_down_object_on_shelf': 9.247614426653692, 'person_puts_down_object_on_table': 1.9884672074906158, 'person_reads_document': 0.7940480628992879, 'person_rides_bicycle': 2.662661823600623, 'person_shakes_hand': 0.7819547332927879, 'person_sits_down': 0.8375202893491961, 'person_stands_up': 1.0285510019795079, 'person_steals_object_from_person': 1.0673909796893626, 'person_talks_on_phone': 0.3031855242664589, 'person_talks_to_person': 0.334895684562076, 'person_texts_on_phone': 0.713951043919232, 'person_transfers_object_to_car': 3.2832615561297605, 'person_transfers_object_to_person': 0.9633429807282274, 'person_unloads_car': 1.1051597100801462, 'vehicle': 1.1953172363332243}
        self._class_to_weight['person_puts_down_object_on_shelf'] = 1.0   # run 5

        self._class_to_index = {'car_drops_off_person': 0, 'car_picks_up_person': 1, 'car_reverses': 2, 'car_starts': 3, 'car_stops': 4, 'car_turns_left': 5, 'car_turns_right': 6, 'hand_interacts_with_person_highfive': 7, 'person': 8, 'person_abandons_object': 9, 'person_carries_heavy_object': 10, 'person_closes_car_door': 11, 'person_closes_car_trunk': 12, 'person_closes_facility_door': 13, 'person_embraces_person': 14, 'person_enters_car': 15, 'person_enters_scene_through_structure': 16, 'person_exits_car': 17, 'person_exits_scene_through_structure': 18, 'person_holds_hand': 19, 'person_interacts_with_laptop': 20, 'person_loads_car': 21, 'person_opens_car_door': 22, 'person_opens_car_trunk': 23, 'person_opens_facility_door': 24, 'person_picks_up_object_from_floor': 25, 'person_picks_up_object_from_table': 26, 'person_purchases_from_cashier': 27, 'person_purchases_from_machine': 28, 'person_puts_down_object_on_floor': 29, 'person_puts_down_object_on_shelf': 30, 'person_puts_down_object_on_table': 31, 'person_reads_document': 32, 'person_rides_bicycle': 33, 'person_shakes_hand': 34, 'person_sits_down': 35, 'person_stands_up': 36, 'person_steals_object_from_person': 37, 'person_talks_on_phone': 38, 'person_talks_to_person': 39, 'person_texts_on_phone': 40, 'person_transfers_object_to_car': 41, 'person_transfers_object_to_person': 42, 'person_unloads_car': 43, 'vehicle': 44}

        self._verb_to_noun = {k:set(['car','vehicle','motorcycle','bus','truck']) if (k.startswith('car') or k.startswith('motorcycle') or k.startswith('vehicle')) else set(['person']) for k in self.classlist()}        
        self._class_to_shortlabel = pycollector.label.pip_to_shortlabel  # FIXME: remove dependency here

        if pretrained:
            self._load_pretrained()
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes())
        elif modelfile is not None:
            self._load_trained(modelfile)
        
    def category(self, x):
        yh = self.forward(x if x.ndim == 5 else torch.unsqueeze(x, 0))
        return [self.index_to_class(int(k)) for (c,k) in zip(*torch.max(yh, dim=1))]

    def category_confidence(self, x):
        yh = self.forward(x if x.ndim == 5 else torch.unsqueeze(x, 0))
        return [(self.index_to_class(int(k)), float(c)) for (c,k) in zip(*torch.max(yh, dim=1))]

    def topk(self, x_logits, k):
        yh = x_logits.detach().cpu().numpy()
        topk = [[(self.index_to_class(j), s[j]) for j in i[-k:][::-1]] for (s,i) in zip(yh, np.argsort(yh, axis=1))]
        return topk

    def topk_probability(self, x_logits, k):
        yh = x_logits.detach().cpu().numpy()
        yh_prob = F.softmax(x_logits, dim=1).detach().cpu().numpy()
        topk = [[(self.index_to_class(j), c[j], p[j]) for j in i[-k:][::-1]] for (c,p,i) in zip(yh, yh_prob, np.argsort(yh, axis=1))]
        return topk
        
    # ---- <LIGHTNING>
    def forward(self, x):
        return self.net(x)  # lighting handles device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_nb, logging=True, valstep=False):
        (x,Y) = batch  
        y_hat = self.forward(x)
        y_hat_softmax = F.softmax(y_hat, dim=1)

        (loss, n_valid, y_validation) = (0, 0, [])
        #C = torch.tensor([self._index_to_training_weight[v] for (k,v) in sorted(self._class_to_index.items(), key=lambda x: x[1])], device=y_hat.device)  # inverse class frequency        
        C = torch.tensor([v for (k,v) in sorted(self._index_to_training_weight.items(), key=lambda x: x[0])], device=y_hat.device)  # inverse class frequency        
        for (yh, yhs, labelstr) in zip(y_hat, y_hat_softmax, Y):
            labels = json.loads(labelstr)
            if labels is None:
                continue  # skip me
            lbllist = [l for lbl in labels for l in lbl]  # list of multi-labels within clip (unpack from JSON to use default collate_fn)
            lbllist = [l for l in lbllist if l in self._class_to_index]  # only allowable classes
            lbl_frequency = vipy.util.countby(lbllist, lambda x: x)  # frequency within clip
            lbl_weight = {k:v/float(len(lbllist)) for (k,v) in lbl_frequency.items()}  # multi-label likelihood within clip, sums to one            
            for (y,w) in lbl_weight.items():
                if valstep:
                    # Pick all labels normalized (https://papers.nips.cc/paper/2019/file/da647c549dde572c2c5edc4f5bef039c-Paper.pdf
                    loss += float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)
                elif self._mlfl:
                    # Pick all labels normalized, with multi-label focal loss
                    loss += torch.min(torch.tensor(1.0, device=y_hat.device), ((w-yhs[self._class_to_index[y]])/w)**2)*float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)
                    
                elif self._mlbl:
                    # Pick all labels normalized with multi-label background loss
                    #j_bg_person = self._class_to_index['person'] if 'person' in self._class_to_index else self._class_to_index['person_walks']  # FIXME: does not generalize
                    #j_bg_vehicle = self._class_to_index['vehicle'] if 'vehicle' in self._class_to_index else self._class_to_index['car_moves']  # FIXME: does not generalize
                    #j = j_bg_person if (y.startswith('person') or y.startswith('hand')) else j_bg_vehicle
                    #loss += ((1-torch.sqrt(yhs[j]*yhs[self._class_to_index[y]]))**2)*float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)
                    raise ('Deprecated')                    
                else:
                    # Pick all labels normalized: https://papers.nips.cc/paper/2019/file/da647c549dde572c2c5edc4f5bef039c-Paper.pdf
                    loss += float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)

            n_valid += 1
            if len(lbllist) > 0:
                y_validation.append( (yh, self._class_to_index[max(lbllist, key=lbllist.count)]) )  # most frequent label in clip
        loss = loss / float(max(1, n_valid))  # batch reduction: mean

        if logging:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss if not valstep else {'loss': loss, 'logit': torch.stack([x for (x,c) in y_validation]), 'classindex': [c for (x,c) in y_validation]}

    def validation_step(self, batch, batch_nb):
        loss = self.training_step(batch, batch_nb, logging=False, valstep=True)['loss']
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)                
        return {'val_loss': avg_loss, 'avg_val_loss': avg_loss}                         
    #---- </LIGHTNING>
    
    @classmethod
    def from_checkpoint(cls, checkpointpath):
        return cls().load_from_checkpoint(checkpointpath)  # lightning
            
    def _load_trained(self, ckptfile):
        self.net = heyvi.model.ResNets_3D_PyTorch.resnet.generate_model(50, n_classes=self.num_classes(), unitnorm=self._unitnorm)
        t = torch.split(self.net.conv1.weight.data, dim=1, split_size_or_sections=1)
        self.net.conv1.weight.data = torch.cat( (*t, t[-1]), dim=1).contiguous()
        self.net.conv1.in_channels = 4  # inflate RGB -> RGBA
        self.load_state_dict(torch.load(ckptfile)['state_dict'])  # FIXME
        self.eval()
        return self
        
    def _load_pretrained(self):

        pthfile = vipy.downloader.downloadif('https://dl.dropboxusercontent.com/s/t3xge6lrfqpklr0/r3d50_kms_200ep.pth',
                                                vipy.util.tocache('r3d50_KMS_200ep.pth'),  # set VIPY_CACHE env 
                                                sha1='39ea626355308d8f75307cab047a8d75862c3261')
        
        net = heyvi.model.ResNets_3D_PyTorch.resnet.generate_model(50, n_classes=1139, unitnorm=self._unitnorm)
        pretrain = torch.load(pthfile, map_location='cpu')
        net.load_state_dict(pretrain['state_dict'])

        # Inflate RGB -> RGBA         
        t = torch.split(net.conv1.weight.data, dim=1, split_size_or_sections=1)
        net.conv1.weight.data = torch.cat( (*t, t[-1]), dim=1).contiguous()
        net.conv1.in_channels = 4

        self.net = net

        return self

    @staticmethod
    def _totensor(v, training, validation, input_size, num_frames, mean, std, noflip=None, show=False, doflip=False):
        assert isinstance(v, vipy.video.Scene), "Invalid input"
        
        try:
            v = v.download() if (not v.isloaded() and not v.hasfilename() and v.hasurl()) else v  # fetch it if necessary, but do not do this during training!        
            if training or validation:
                random.seed()  # force randomness after fork() 
                (ai,aj) = (v.primary_activity().startframe(), v.primary_activity().endframe())  # activity (start,end)
                (ti,tj) = (v.actor().startframe(), v.actor().endframe())  # track (start,end) 
                startframe = random.randint(max(0, ti-(num_frames//2)), max(1, tj-(num_frames//2)))  # random startframe that contains track
                endframe = min((startframe+num_frames), aj)  # endframe truncated to be end of activity
                (startframe, endframe) = (startframe, endframe) if (startframe < endframe) else (max(0, aj-num_frames), aj)  # fallback
                assert endframe - startframe <= num_frames
                vc = v.clone().clip(startframe, endframe)    # may fail for some short clips
                vc = vc.trackcrop(dilate=1.2, maxsquare=True)  # may be None if clip contains no track
                vc = vc.resize(input_size, input_size)   
                vc = vc.fliplr() if (doflip or (random.random() > 0.5)) and (noflip is None or vc.category() not in noflip) else vc
            else:
                vc = v.trackcrop(dilate=1.2, maxsquare=True)  # may be None if clip contains no track
                vc = vc.resize(input_size, input_size)  # TESTING: this may introduce a preview()
                vc = vc.fliplr() if doflip and (noflip is None or vc.category() not in noflip) else vc
                
            if show:
                vc.clone().resize(512,512).show(timestamp=True)
                vc.clone().binarymask().frame(0).rgb().show(figure='binary mask: frame 0')
                
            vc = vc.load(shape=(input_size, input_size, 3)).normalize(mean=mean, std=std, scale=1.0/255.0)  # [0,255] -> [0,1], triggers load() with known shape
            (t,lbl) = vc.torch(startframe=0, length=num_frames, boundary='cyclic', order='cdhw', withlabel=training or validation, nonelabel=True)  # (c=3)x(d=num_frames)x(H=input_size)x(W=input_size), reuses vc._array
            t = torch.cat((t, vc.asfloatmask(fg=0.5, bg=-0.5).torch(startframe=0, length=num_frames, boundary='cyclic', order='cdhw')), dim=0)  # (c=4) x (d=num_frames) x (H=input_size) x (W=input_size), copy
            
        except Exception as e:
            if training or validation:
                #print('ERROR: %s' % (str(v)))
                t = torch.zeros(4, num_frames, input_size, input_size)  # skip me
                lbl = None
            else:
                print('WARNING: discarding tensor for video "%s" with exception "%s"' % (str(v), str(e)))
                t = torch.zeros(4, num_frames, input_size, input_size)  # skip me (should never get here)
            
        if training or validation:
            return (t, json.dumps(lbl))  # json to use default collate_fn
        else:
            return t

    def totensor(self, v=None, training=False, validation=False, show=False, doflip=False):
        """Return captured lambda function if v=None, else return tensor"""    
        assert v is None or isinstance(v, vipy.video.Scene), "Invalid input"
        f = (lambda v, num_frames=self._num_frames, input_size=self._input_size, mean=self._mean, std=self._std, training=training, validation=validation, show=show:
             PIP_250k._totensor(v, training, validation, input_size, num_frames, mean, std, noflip=['car_turns_left', 'car_turns_right'], show=show, doflip=doflip))
        return f(v) if v is not None else f
    


class PIP_370k(PIP_250k, pl.LightningModule, ActivityRecognition):

    def __init__(self, pretrained=True, deterministic=False, modelfile=None, mlbl=False, mlfl=True, unitnorm=False):
        pl.LightningModule.__init__(self)
        ActivityRecognition.__init__(self)  

        self._input_size = 112
        self._num_frames = 16        
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._mlfl = mlfl
        self._mlbl = mlbl
        self._calibrated = False
        self._calibrated_constant = -1.5
        self._unitnorm = unitnorm

        if deterministic:
            np.random.seed(42)
        
        # Generated using vipy.dataset.Dataset(...).multilabel_inverse_frequency_weight()
        self._class_to_training_weight = {'car_drops_off_person': 0.7858124882763793, 'car_moves': 0.18439798528529147, 'car_picks_up_person': 0.7380666753394193, 'car_reverses': 0.5753369570213479, 'car_starts': 0.47486292483745757, 'car_stops': 0.44244800737774037, 'car_turns_left': 0.7697107319736983, 'car_turns_right': 0.5412936796835607, 'hand_interacts_with_person': 0.2794031245117859, 'person_abandons_package': 1.0789960714517162, 'person_carries_heavy_object': 0.5032333530901552, 'person_closes_car_door': 0.46460114438995603, 'person_closes_car_trunk': 0.6824201392305784, 'person_closes_facility_door': 0.38990434394080076, 'person_embraces_person': 0.6457437695527715, 'person_enters_car': 0.6934926810021877, 'person_enters_scene_through_structure': 0.2586965095740063, 'person_exits_car': 0.6766386632434479, 'person_exits_scene_through_structure': 0.33054895987676847, 'person_interacts_with_laptop': 0.6720176496986436, 'person_loads_car': 0.6880555743488312, 'person_opens_car_door': 0.4069868136393968, 'person_opens_car_trunk': 0.6911966903970317, 'person_opens_facility_door': 0.3018924474724252, 'person_picks_up_object': 0.4298381074082487, 'person_purchases_from_cashier': 5.479834409621331, 'person_purchases_from_machine': 5.31528236654537, 'person_puts_down_object': 0.2804690906037155, 'person_reads_document': 0.5476186269530937, 'person_rides_bicycle': 1.6090962879286763, 'person_sits_down': 0.4750148103149501, 'person_stands_up': 0.5022364750834624, 'person_steals_object': 0.910991409921711, 'person_talks_on_phone': 0.15771902851484076, 'person_talks_to_person': 0.21362675034201736, 'person_texts_on_phone': 0.3328378404741194, 'person_transfers_object_to_car': 2.964890512157848, 'person_transfers_object_to_person': 0.6481292773603928, 'person_unloads_car': 0.515379337544623, 'person_walks': 6.341278284010202}
        self._class_to_weight = self._class_to_training_weight  # backwards compatibility

        # Generated using vipy.dataset.Dataset(...).class_to_index()
        self._class_to_index = {'car_drops_off_person': 0, 'car_moves': 1, 'car_picks_up_person': 2, 'car_reverses': 3, 'car_starts': 4, 'car_stops': 5, 'car_turns_left': 6, 'car_turns_right': 7, 'hand_interacts_with_person': 8, 'person_abandons_package': 9, 'person_carries_heavy_object': 10, 'person_closes_car_door': 11, 'person_closes_car_trunk': 12, 'person_closes_facility_door': 13, 'person_embraces_person': 14, 'person_enters_car': 15, 'person_enters_scene_through_structure': 16, 'person_exits_car': 17, 'person_exits_scene_through_structure': 18, 'person_interacts_with_laptop': 19, 'person_loads_car': 20, 'person_opens_car_door': 21, 'person_opens_car_trunk': 22, 'person_opens_facility_door': 23, 'person_picks_up_object': 24, 'person_purchases_from_cashier': 25, 'person_purchases_from_machine': 26, 'person_puts_down_object': 27, 'person_reads_document': 28, 'person_rides_bicycle': 29, 'person_sits_down': 30, 'person_stands_up': 31, 'person_steals_object': 32, 'person_talks_on_phone': 33, 'person_talks_to_person': 34, 'person_texts_on_phone': 35, 'person_transfers_object_to_car': 36, 'person_transfers_object_to_person': 37, 'person_unloads_car': 38, 'person_walks': 39}
        
        self._index_to_training_weight = {self._class_to_index[k]:v for (k,v) in self._class_to_weight.items()}

        self._verb_to_noun = {k:set(['car','vehicle','motorcycle','bus','truck']) if (k.startswith('car') or k.startswith('motorcycle') or k.startswith('vehicle')) else set(['person']) for k in self.classlist()}        
        self._class_to_shortlabel = heyvi.label.pip_to_shortlabel
        self._class_to_shortlabel.update( vipy.data.meva.d_category_to_shortlabel )

        if pretrained:
            self._load_pretrained()
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes())
        elif modelfile is not None:
            self._load_trained(modelfile)

    def topk(self, x, k=None):
        """Return the top-k classes for a 3 second activity proposal along with framewise ground truth"""        
        yh = self.forward(x if x.ndim == 5 else x.unsqueeze(0)).detach().cpu().numpy()
        k = k if k is not None else self.num_classes()
        return [ [self.index_to_class(int(j)) for j in i[-k:][::-1]] for (s,i) in zip(yh, np.argsort(yh, axis=1))]
            
    @staticmethod
    def _totensor(v, training, validation, input_size, num_frames, mean, std, noflip=None, show=False, doflip=False, stride_jitter=3, asjson=False, classname='heyvi.recognition.PIP_370k'):
        assert isinstance(v, vipy.video.Scene), "Invalid input"
        
        try:
            v = v.download() if (not v.isloaded() and v.hasurl() and not v.hasfilename()) else v  # fetch it if necessary, but do not do this during training!        
            if training or validation:
                vc = v.clone()  # these frames are shared across all tracks
                random.seed()  # force randomness after fork() 
                (clipstart, clipend) = vc.cliprange()  # clip (start, end) relative to video 
                (clipstart, clipend) = (clipstart if clipstart is not None else 0,   
                                        clipend if clipend is not None else int(np.floor(v.duration_in_frames_of_videofile() * (vc.framerate() / v.framerate_of_videofile()))))  # (yuck)
                # WARNINGS: 
                # - There exist videos with tracks outside the image rectangle due to the padding in stabilization.  
                # - There exist MEVA videos that have no tracks at the beginning and end of the padded clip since the annotations only exist for the activity
                # - There exist MEVA videos with activities that are longer than the tracks, if so, keep the interval of the activity that contains the track
                # - There exist MEVA videos with multiple objects, need to include only primary actor
                
                # - turning activities may be outside the frame (filter these)
                # - turning activities may turn into the stabilized black area.  Is this avoidaable?
                # - all of the training activities should be centered on the activity.  See if not.
                
                if (clipend - clipstart) > (num_frames + stride_jitter):
                    a = vc.primary_activity().clone().padto(num_frames/float(vc.framerate()))  # for context only, may be past end of clip now!
                    (ai, aj) = (a.startframe(), a.endframe())  # activity (start,end) relative to (clipstart, clipend)
                    (ai, aj) = (max(ai, vc.actor().startframe()), min(aj, vc.actor().endframe()))  # clip activity to when actor is present
                    startframe = random.randint(ai, aj-num_frames-1) if aj-num_frames-1 > ai else ai
                    startframe = max(0, startframe + random.randint(-stride_jitter, stride_jitter))   # +/- 3 frames jitter for activity stride
                    endframe = min(clipend-clipstart-1, startframe + num_frames)  # new end cannot be past duration of clip
                    if (endframe > startframe) and ((endframe - startframe) < (clipend - clipstart)):
                        vc = vc.clip(startframe, endframe)
                    else: 
                        raise ValueError('invalid clip for "%s"' % str(v))
                vc = vc.trackcrop(dilate=1.2, maxsquare=True)  # may be None if clip contains no track
                vc = vc.resize(input_size, input_size)   
                vc = vc.fliplr() if (doflip or (random.random() > 0.5)) and (noflip is None or vc.category() not in noflip) else vc
            else:
                vc = v.clone(shallow=True)  # shallow clone, trackcrop() will result in new array and v will be preserved
                vc = vc.trackcrop(dilate=1.2, maxsquare=True)  # may be None if clip contains no track
                vc = vc.resize(input_size, input_size)  # This may introduce a preview()
                vc = vc.fliplr() if doflip and (noflip is None or vc.category() not in noflip) else vc
                
            if show:
                vc.clone().resize(512,512).show(timestamp=True)
                vc.clone().binarymask().frame(0).gain(255).rgb().show(figure='binary mask: frame 0')
                
            vc = vc.load(shape=(input_size, input_size, 3)).normalize(mean=mean, std=std, scale=1.0/255.0)  # [0,255] -> [0,1], triggers load() with known shape
            (t,lbl) = vc.torch(startframe=0, length=num_frames, boundary='repeat', order='cdhw', withlabel=training or validation, nonelabel=True)  # (c=3)x(d=num_frames)x(H=input_size)x(W=input_size), reuses vc._array
            t = torch.cat((t, vc.asfloatmask(fg=0.5, bg=-0.5).torch(startframe=0, length=num_frames, boundary='repeat', order='cdhw')), dim=0)  # (c=4) x (d=num_frames) x (H=input_size) x (W=input_size), copy

        except Exception as e:
            if training or validation:
                print('[heyvi.recognition.%s._totensor][SKIPPING]: video="%s", exception="%s"' % (classname, str(vc), str(e)))
                (t, lbl) = (torch.zeros(4, num_frames, input_size, input_size), None)  # must always return conformal tensor (label=None means it will be ignored)
            else:
                print('[heyvi.recognition.%s._totensor][ERROR]: discarding tensor for video "%s" with exception "%s"' % (classname, str(vc), str(e)))
                #t = torch.zeros(4, num_frames, input_size, input_size)  # skip me (should never get here)
                raise

        if training or validation:
            return (t, json.dumps(lbl) if not asjson else lbl)  # json to use default torch collate_fn
        else:
            return t

    def totensor(self, v=None, training=False, validation=False, show=False, doflip=False, asjson=False):
        """Return captured lambda function if v=None, else return tensor"""
        assert v is None or isinstance(v, vipy.video.Scene), "Invalid input"
        f = (lambda v, num_frames=self._num_frames, input_size=self._input_size, mean=self._mean, std=self._std, training=training, validation=validation, show=show, classname=self.__class__.__name__:
             PIP_370k._totensor(v, training, validation, input_size, num_frames, mean, std, noflip=['car_turns_left', 'car_turns_right', 'vehicle_turns_left', 'vehicle_turns_right', 'motorcycle_turns_left', 'motorcycle_turns_right'], show=show, doflip=doflip, asjson=asjson, classname=classname))
        return f(v) if v is not None else f


class CAP(PIP_370k, pl.LightningModule, ActivityRecognition):
    def __init__(self, modelfile=None, deterministic=False, pretrained=None, mlbl=None, mlfl=True, calibrated_constant=-1.5, calibrated=False, unitnorm=False):
        pl.LightningModule.__init__(self)
        ActivityRecognition.__init__(self)  

        self._input_size = 112
        self._num_frames = 16        
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._mlfl = True
        self._mlbl = False
        self._calibrated_constant = calibrated_constant
        self._calibrated = calibrated
        self._unitnorm = unitnorm

        if deterministic:
            np.random.seed(42)

        version = 3
        if version == 1:
            print('[heyvi.recognition.CAP]: version == 1')  # cap_l2norm_e23s96095.ckpt and earlier

            # Generated using vipy.dataset.Dataset.multilabel_inverse_frequency_weight()
            # - WARNING: under-represented classes are truncated at a maximum weight of one
            # - python 3.7 can use importlib.resources
            self._class_to_training_weight = {k:float(v) for (k,v) in vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'class_to_training_weight.csv'))}
            self._class_to_weight = self._class_to_training_weight  # backwards compatibility
            
            # Generated using vipy.dataset.Dataset.class_to_index()
            self._class_to_index = {k:int(v) for (k,v) in vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'class.csv'))}
            self._index_to_training_weight = {self._class_to_index[k]:v for (k,v) in self._class_to_weight.items()}
            
            # Generated using vipy.dataset.Dataset.class_to_shortlabel()
            self._class_to_shortlabel = dict(vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'class_to_shortlabel.csv')))
            self._class_to_shortlabel.update( vipy.data.meva.d_category_to_shortlabel )

        elif version == 2:
            print('[heyvi.recognition.CAP]: version==2')

            # Generated using vipy.dataset.Dataset.multilabel_inverse_frequency_weight()
            # - WARNING: under-represented classes are truncated at a maximum weight of one
            self._index_to_training_weight = {int(k):float(v) for (k,v) in vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'coarse_index_to_training_weight.csv'))}

            # Generated using vipy.dataset.Dataset.class_to_index()
            self._class_to_index = {k:int(v) for (k,v) in vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'coarse_class_to_index.csv'))}
            self._index_to_class = {int(k):v for (k,v) in vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'coarse_index_to_class.csv'))}

            # Derived
            self._class_to_training_weight = {k:self._index_to_training_weight[v] for (k,v) in self._class_to_index.items()}
            self._class_to_weight = self._class_to_training_weight  # backwards compatibility

        elif version == 3:
            print('[heyvi.recognition.CAP]: version==3')

            # Generated using vipy.dataset.Dataset.multilabel_inverse_frequency_weight()
            # - WARNING: under-represented classes are truncated at a maximum weight of one
            self._index_to_training_weight = {int(k):float(v) for (k,v) in vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'background_index_to_training_weight.csv'))}

            # Generated using vipy.dataset.Dataset.class_to_index()
            self._class_to_index = {k:int(v) for (k,v) in vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'background_class_to_index.csv'))}
            self._index_to_class = {int(k):v for (k,v) in vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'background_index_to_class.csv'))}

            # Derived
            self._class_to_training_weight = {k:self._index_to_training_weight[v] for (k,v) in self._class_to_index.items()}
            self._class_to_weight = self._class_to_training_weight  # backwards compatibility

        else:
            raise
            
        # Generated using vipy.dataset.Dataset.class_to_shortlabel()
        self._class_to_shortlabel = dict(vipy.util.readcsv(os.path.join(os.path.dirname(heyvi.__file__), 'model', 'cap', 'class_to_shortlabel.csv')))
        self._class_to_shortlabel.update( vipy.data.meva.d_category_to_shortlabel )


        # Calibration state: trained at validation epoch end
        if self._calibrated:
            self.register_buffer('_calibration_multiclass', torch.zeros(1,1))
            self.register_buffer('_calibration_binary', torch.zeros(3,self.num_classes()))

        if modelfile is not None:
            self._load_trained(modelfile)
        else:
            self._load_pretrained()
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes())

    
    #---- <LIGHTNING>
    def forward(self, x):
        return self.net(x)  

    def validation_step(self, batch, batch_nb):
        s = self.training_step(batch, batch_nb, logging=False, valstep=True)
        self.log('val_loss', s['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'val_loss': s['loss'], 'logit': s['logit'], 'classindex':s['classindex']}

    def validation_step_end(self, outputs):
        return {'val_loss': outputs['val_loss'],  
                'logit': outputs['logit'],  # accumulated across GPUs (single machine multi-GPU only)
                'classindex': outputs['classindex']}

    def validation_epoch_end(self, outputs):        
        avg_loss = torch.cat([x['val_loss'].flatten() for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)  # for checkpointing

        # Calibration: will be saved as registered buffer in checkpoint for calibration
        if self._calibrated and self.trainer.is_global_zero:
            from netcal.scaling import LogisticCalibration, TemperatureScaling
            logits = torch.stack([x for output in outputs for x in output['logit']]).detach().cpu().numpy()
            ground_truth = torch.cat([x.flatten() for output in outputs for x in output['classindex']]).flatten().detach().cpu().numpy()
            multiclass = TemperatureScaling()
            multiclass.fit(F.softmax(torch.from_numpy(logits), dim=1).cpu().numpy(), ground_truth)
            binary = {k:(LogisticCalibration(), float(np.mean(logits[:,k]))) for k in sorted(self.class_to_index().values())}
            for (k,(b,m)) in binary.items():
                (binary_confidences, binary_ground_truth) = (torch.sigmoid(torch.from_numpy(logits[:,k]-m).flatten()).cpu().numpy(), np.array([1 if y==k else 0 for y in ground_truth]))
                if np.any(binary_ground_truth):
                    b.fit(binary_confidences, binary_ground_truth)
                else:
                    binary[k] = (None,0)  # no samples for calibration, prediction will always be zero

            with torch.no_grad():
                calibration_multiclass = torch.tensor(float(multiclass._sites['weights']['values']))
                calibration_binary = torch.stack((torch.tensor([float(b._sites['weights']['values']) if b is not None else 0 for (k,(b,m)) in binary.items()]), 
                                                  torch.tensor([float(b._sites['bias']['values']) if b is not None else 0 for (k,(b,m)) in binary.items()]), 
                                                  torch.tensor([float(m) if b is not None else 0 for (k,(b,m)) in binary.items()])))
            
                self._calibration_multiclass[:] = calibration_multiclass.to(device=torch.device('cuda:0'))
                self._calibration_binary[:] = calibration_binary.to(device=torch.device('cuda:0'))
    #---- </LIGHTNING>
        
    def totensor(self, v=None, training=False, validation=False, show=False, doflip=False, asjson=False):
        """Return captured lambda function if v=None, else return tensor"""    
        assert v is None or isinstance(v, vipy.video.Scene), "Invalid input"
        f = (lambda v, num_frames=self._num_frames, input_size=self._input_size, mean=self._mean, std=self._std, training=training, validation=validation, show=show, classname=self.__class__.__name__, doflip=doflip, asjson=asjson:
             PIP_370k._totensor(v, training, validation, input_size, num_frames, mean, std, noflip=['car_turns_left', 'car_turns_right', 'vehicle_turns_left', 'vehicle_turns_right', 'motorcycle_turns_left', 'motorcycle_turns_right'], show=show, doflip=doflip, asjson=asjson, classname=classname))
        return f(v) if v is not None else f

    def calibration(self, x_logits):
        assert torch.is_tensor(self._calibration_multiclass) and self._calibration_multiclass.shape == (1,1)
        assert torch.is_tensor(self._calibration_binary) and self._calibration_binary.shape == (3, self.num_classes())
        (n, T, (w,b,o), eps) = (self.num_classes(), self._calibration_multiclass, self._calibration_binary, np.finfo(np.float64).eps)  # (TemperatureScaling, PlattScaling=(weight, bias, offset))

        lr = torch.from_numpy(np.multiply(1.0/(1+np.exp(-(np.multiply(np.array(w*0 + 1).reshape(1,n).astype(np.float64), scipy.special.logit(np.clip(torch.sigmoid(x_logits-o.view(1,n)).detach().cpu().numpy().astype(np.float64), eps, 1-eps))))+np.array(b).reshape(1,n).astype(np.float64))), (np.array(o)!=0).reshape(1,n)).astype(np.float32))
        sm = F.softmax(torch.log(torch.clamp(F.softmax(x_logits, dim=1), eps, 1-eps)) / T, dim=1)
        
        #sm = F.softmax(x_logits / T, dim=1)  # temperature only
        #lr = torch.multiply(torch.sigmoid(x_logits-o.view(1,n)+b.view(1,n)), (o!=0).view(1,n))   # bias only
        return torch.multiply(sm, lr)

        
class ActivityTracker(PIP_370k):
    """Video Activity detection.
        
    Args (__call__):
        vi [generator of `vipy.video.Scene`]:  The input video to be updated in place with detections.  This is a generator which is output from heyvi.detection.MultiscaleVideoTracker.__call__
        activityiou [float]: The minimum temporal iou for activity assignment
        mirror [bool]:  If true, encode using the mean of a video encoding and the mirrored video encoding.  This is slower as it requires 2x GPU forward passes
        minprob [float]: The minimum probability for new activity detection
        trackconf [float]: The minimum object detection confidence for new tracks
        maxdets [int]:  The maximum number of allowable detections per frame.  If there are more detections per frame tha maxdets, sort them by confidence and use only the top maxdets best
        avgdets [int]:  The number of allowable detections per frame if throttled
        buffered [bool]:  If true, then buffer streams.  This is useful for activity detection on live streams.            
        finalized [bool, int]:  If False then do not finalize(), If True finalize() only at the end, If int, then finalize every int frames.  This is useful for streaming activity detection on unbounded inputs. 
        
    Returns:
        The input video is updated in place.
    
    """    
    def __init__(self, stride=3, activities=None, gpus=None, batchsize=None, mlbl=False, mlfl=True, modelfile=None):
        assert modelfile is not None, "Contact <info@visym.com> for access to non-public model files"

        super().__init__(pretrained=False, modelfile=modelfile, mlbl=mlbl, mlfl=mlfl)
        self._stride = stride
        self._allowable_activities = {k:v for (k,v) in [(a,a) if not isinstance(a, tuple) else a for a in activities]} if activities is not None else {k:k for k in self.classlist()}
        self._verb_to_noun = {k:set(['car','vehicle','motorcycle','bus','truck']) if (k.startswith('car') or k.startswith('motorcycle') or k.startswith('vehicle')) else set(['person']) for k in self.classlist()+list(self._allowable_activities.values())+list(self._allowable_activities.keys())}  # HACK: fixme
        self._batchsize_per_gpu = batchsize
        self._gpus = gpus

        if gpus is not None:
            assert torch.cuda.is_available()
            assert batchsize is not None
            self._devices = ['cuda:%d' % k for k in gpus]
            self._gpus = [copy.deepcopy(self.net).to(d, non_blocking=False) for d in self._devices]  
            for m in self._gpus:
                m.eval()
        torch.set_grad_enabled(False)

        self._logit_pooling = None  # cache
        
    def temporal_stride(self, s=None):
        if s is not None:
            self._stride = s
            return self
        else:
            return self._stride

    def forward(self, x):
        """Overload forward for multi-gpu batch.  Don't use torch DataParallel!"""
        if self._gpus is None:
            return super().forward(x)  # cpu
        else:
            x_forward = None            
            for b in x.pin_memory().split(self._batchsize_per_gpu*len(self._gpus)):  # pinned copy
                n_todevice = np.sum(np.array([1 if k<len(b) else 0 for k in range(int(len(self._devices)*np.ceil(len(b)/len(self._devices))))]).reshape(-1, len(self._devices)), axis=0).tolist()
                todevice = [t.to(d, non_blocking=True) for (t,d) in zip(b.split(n_todevice), self._devices) if len(t)>0]   # async device copy
                ondevice = [m(t) for (m,t) in zip(self._gpus, todevice)]   # async
                fromdevice = torch.cat([t.cpu() for t in ondevice], dim=0)
                x_forward = fromdevice if x_forward is None else torch.cat((x_forward, fromdevice), dim=0)
                del ondevice, todevice, fromdevice, b  # force garbage collection of GPU memory
            del x  # force garbage collection
            return x_forward

    def lrt(self, x_logits, lrt_threshold=None):
        """top-k with likelihood ratio test with background null hypothesis"""
        j_bg_person = self._class_to_index['person'] if 'person' in self._class_to_index else self._class_to_index['person_walks']  # FIXME
        j_bg_vehicle = self._class_to_index['vehicle'] if 'vehicle' in self._class_to_index else self._class_to_index['car_moves']  # FIXME

        yh = x_logits.detach().cpu().numpy()
        yh_softmax = F.softmax(x_logits, dim=1).detach().cpu()
        p_null = np.maximum(yh[:, j_bg_person], yh[:, j_bg_vehicle]).reshape(yh.shape[0], 1)
        lr = yh - p_null   # ~= log likelihood ratio
        f_logistic = lambda x,b,s=1.0: float(1.0 / (1.0 + np.exp(-s*(x + b))))
        return [sorted([(self.index_to_class(j), float(s[j]), float(t[j]), f_logistic(s[j], 1.0)*f_logistic(t[j], 0.0), float(sm[j])) for j in range(len(s)) if (lrt_threshold is None or t[j] >= lrt_threshold)], key=lambda x: x[3], reverse=True) for (s,t,sm) in zip(yh, lr, yh_softmax)]

    def logit_pooling(self, x_logits, classmap):
        if self._logit_pooling is None:
            assert isinstance(classmap, dict)
            index_to_class = {k:c for (k,c) in enumerate(set([k if v is None else v for (k,v) in classmap.items()]))}  # target class index
            class_to_index = {v:k for (k,v) in index_to_class.items()}  # target class to index
            class_to_group = [[self.class_to_index(c[0]) for c in v if c[0] in self.class_to_index()] for (k,v) in sorted(vipy.util.groupbyasdict([(k,k) if v is None else (k,v) for (k,v) in classmap.items()], lambda x: x[1]).items(), key=lambda x: class_to_index[x[0]])]  # target class to source index
            self._logit_pooling = {'index_to_class': index_to_class, 'class_to_group': class_to_group}
        (index_to_class, class_to_group) = (self._logit_pooling['index_to_class'], self._logit_pooling['class_to_group'])  # cached
        yh = torch.tensor([max(r[j]) if len(r[j])>0 else (min(r)) for (i,r) in enumerate(x_logits) for j in class_to_group]).reshape(len(x_logits), len(index_to_class)).detach()
        yh_softmax = F.softmax(yh, dim=1).detach().cpu()
        return [[(index_to_class[j], float(sm[j]), float(s[j])) for j in range(len(sm))] for (s,sm) in zip(yh.cpu(), yh_softmax)]
        
    def softmax(self, x_logits):
        """Return a list of lists [(class_label, float(softmax), float(logit) ... ] for all classes and batches"""
        yh = x_logits.detach().cpu().numpy()
        yh_softmax = F.softmax(x_logits, dim=1).detach().cpu()
        d = self.index_to_class()
        if not self._calibrated:
            return [[(d[j], float(sm[j]), float(s[j])) for j in range(len(sm))] for (s,sm) in zip(yh, yh_softmax)]
        else:
            yh_softmax = self.calibration(x_logits)
            return [[(d[j], float(sm[j]), float(s[j])) for j in range(len(sm))] for (s,sm) in zip(yh, yh_softmax)]            

    def finalize(self, vo, trackconf=None, activityconf=None, startframe=None, endframe=None, mintracklen=None):
        """In place filtering of video to finalize"""
        assert isinstance(vo, vipy.video.Scene)

        tofinalize = set([ai for (ai,a) in vo.activities().items() if (endframe is None or a.endframe() <= endframe) and (startframe is None or a.endframe() >= startframe)])
        tofinalize = tofinalize.union([ti for (ti,t) in vo.tracks().items() if ((endframe is None or t.endframe() <= endframe) and (startframe is None or t.endframe() >= startframe)) or any([ti == vo.activities(id=ai).actorid() for ai in tofinalize])])

        # Bad tracks:  Remove low confidence or too short non-moving tracks, and associated activities
        # - will throw exception that 'vo referenced before assignment' if one loop did not succceed
        if trackconf is not None:
            vo.trackfilter(lambda t: t.id() not in tofinalize or len(t)>=(mintracklen if mintracklen is not None else vo.framerate()) and (t.confidence() >= trackconf or t.startbox().iou(t.endbox()) == 0)).activityfilter(lambda a: a.id() not in tofinalize or a.actorid() in vo.tracks())  
        
        # Activity probability:  noun_probability*verb probability
        nounconf = {k:t.confidence(samples=8) for (k,t) in vo.tracks().items() if t.id() in tofinalize}   # 
        vo.activitymap(lambda a: a.confidence(nounconf[a.actorid()]*a.confidence()) if a.id() in tofinalize else a)
        
        # Missing objects:  Significantly reduce confidence of complex classes (yuck)
        vo.activitymap(lambda a: a.confidence(0.01*a.confidence()) if (a.id() in tofinalize and a.category() in ['person_purchases']) else a) 
        
        # Vehicle turns:  High confidence vehicle turns must be a minimum angle
        vo.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.id() in tofinalize and
                                                                      (a.category() in ['vehicle_turns_left', 'vehicle_turns_right']) and
                                                                      (abs(vo.track(a.actorid()).bearing_change(a.startframe(), a.endframe(), dt=vo.framerate(), samples=5)) < (np.pi/16))) else a) 
        
        # Vehicle turns:  U-turn can only be distinguished from left/right turn at the end of a track by looking at the turn angle
        vo.activitymap(lambda a: a.category('vehicle_makes_u_turn').shortlabel('u turn') if (a.id() in tofinalize and
                                                                                             (a.category() in ['vehicle_turns_left', 'vehicle_turns_right']) and
                                                                                             (abs(vo.track(a.actorid()).bearing_change(a.startframe(), a.endframe(), dt=vo.framerate(), samples=5)) > (np.pi-(np.pi/2)))) else a)
        
        # Background activities:  Use logistic confidence on logit due to lack of background class "person stands", otherwise every standing person is using a phone
        if self._calibrated_constant is not None:            
            f_logistic = lambda x,b,s=1.0: float(1.0 / (1.0 + np.exp(-s*(x + b))))
            vo.activitymap(lambda a: a.confidence(a.confidence()*f_logistic(a.attributes['logit'], self._calibrated_constant)) if a.id() in tofinalize else a)  
            
        # Complex activities: remove steal/abandon and replace with picks up / puts down
        vo.activityfilter(lambda a: a.category() not in ['person_steals_object', 'person_abandons_package'])
        newlist = [vo.add(vipy.activity.Activity(startframe=a.startframe(), endframe=a.endframe(), category='person_steals_object', shortlabel='steals', confidence=0.5*a.confidence(), framerate=vo.framerate(), actorid=a.actorid(), attributes={'pip':'person_picks_up_object'}))
                   for a in vo.activitylist() if a.category() == 'person_picks_up_object']
        newlist = [vo.add(vipy.activity.Activity(startframe=a.startframe(), endframe=a.endframe(), category='person_abandons_package', shortlabel='abandons', confidence=0.5*a.confidence(), framerate=vo.framerate(), actorid=a.actorid(), attributes={'pip':'person_puts_down_object'}))
                   for a in vo.activitylist() if a.category() == 'person_puts_down_object']
            
        # Vehicle/person interaction: 'vehicle_drops_off_person'/'vehicle_picks_up_person'  must be followed by car driving away/pulling up, must be accompanied by person track start/end
        vo.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.id() in tofinalize and
                                                                      a.category() == 'vehicle_drops_off_person' and
                                                                      (not vo.track(a.actorid()).ismoving(a.middleframe(), a.endframe()+10*vo.framerate()) or
                                                                       not any([t.category() == 'person' and t.segment_maxiou(vo.track(a._actorid), t.startframe(), t.startframe()+1) > 0 for t in vo.tracks().values()]))) else a)
        vo.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.id() in tofinalize and
                                                                      a.category() == 'vehicle_picks_up_person' and
                                                                      (not vo.track(a.actorid()).ismoving(a.startframe()-10*vo.framerate(), a.middleframe()) or
                                                                       not any([t.category() == 'person' and t.segment_maxiou(vo.track(a._actorid), t.endframe()-1, t.endframe()) > 0 for t in vo.tracks().values()]))) else a)
        
        # Person/Bicycle track: riding must be accompanied by an associated moving bicycle track
        vo.activityfilter(lambda a: a.id() not in tofinalize or a.category() != 'person_rides_bicycle')
        bikelist = [vo.add(vipy.activity.Activity(startframe=t.startframe(), endframe=t.endframe(), category='person_rides_bicycle', shortlabel='rides', confidence=t.confidence(samples=8), framerate=vo.framerate(), actorid=t.id(), attributes={'pip':'person_rides_bicycle'}))
                    for (tk,t) in vo.tracks().items() if (t.id() in tofinalize and t.category() == 'bicycle' and t.ismoving())]
        
        # Person/Vehicle track: person/vehicle interaction must be accompanied by an associated stopped vehicle track
        dstbox = {k:vo.track(a.actorid()).boundingbox(a.startframe(), a.endframe()) for (k,a) in vo.activities().items() if (a.id() in tofinalize and a.category().startswith('person') and ('vehicle' in a.category() or 'trunk' in a.category()))}  # precompute            
        vo.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.id() in tofinalize and
                                                                      (a.category().startswith('person') and ('vehicle' in a.category() or 'trunk' in a.category())) and
                                                                      not any([t.category() == 'vehicle' and 
                                                                               t.during(a.startframe()) and
                                                                               not t.ismoving(a.startframe(), a.endframe()) and
                                                                               t[a.startframe()].hasintersection(dstbox[a._id])
                                                                               for t in vo.tracks().values()])) else a)
        
        # Vehicle/Person track: vehicle/person interaction must be accompanied by an associated person track
        vo.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.id() in tofinalize and (a.category().startswith('vehicle') and ('person' in a.category())) and not any([t.category() == 'person' and t.segment_maxiou(vo.track(a._actorid), a._startframe, a._endframe) > 0 for t in vo.tracks().values()])) else a)
        
        # Person track: enter/exit scene cannot be at the image boundary
        boundary = vo.framebox().dilate(0.9)
        vo.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.id() in tofinalize and a.category() == 'person_enters_scene_through_structure' and vo.track(a.actorid())[max(a.startframe(), vo.track(a.actorid()).startframe())].cover(boundary) < 1) else a)
        vo.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.id() in tofinalize and a.category() == 'person_exits_scene_through_structure' and vo.track(a.actorid())[min(a.endframe(), vo.track(a.actorid()).endframe())].cover(boundary) < 1) else a)
        
        # Activity union:  Temporal gaps less than support should be merged into one activity detection for a single track
        # Activity union:  "Brief" breaks (<5 seconds) of confident activities should be merged into one activity detection for a single track
        briefmerge = set(['person_reads_document', 'person_interacts_with_laptop', 'person_talks_to_person', 'person_purchases', 'person_steals_object', 'person_talks_on_phone', 'person_texts_on_phone', 'person_rides_bicycle', 'person_carries_heavy_object', 'person', 'person_walks', 'vehicle', 'car_moves'])  
        merged = set([])
        mergeable_dets = [a for a in vo.activities().values() if a.id() in tofinalize and a.confidence() > 0.2]  # only mergeable detections
        mergeable_dets.sort(key=lambda a: a.startframe())  # in-place
        for a in mergeable_dets:
            for o in mergeable_dets:
                if ((o._startframe >= a._startframe) and (a._id != o._id) and (o._actorid == a._actorid) and (o._label == a._label) and (o._id not in merged) and (a._id not in merged) and
                    ((a.temporal_distance(o) <= self.temporal_support() or (a.category() in briefmerge and a.temporal_distance(o) < 5*vo.framerate())))):
                    a.union(o)  # in-place update
                    merged.add(o.id())
        vo.activityfilter(lambda a: a.id() not in tofinalize or a.id() not in merged)

        # Group activity: Must be accompanied by a friend with the same activity detection
        categories = ['person_embraces_person', 'hand_interacts_with_person', 'person_talks_to_person', 'person_transfers_object']           
        dstbox = {k:vo.track(a.actorid()).boundingbox(a.startframe(), a.endframe()) for (k,a) in vo.activities().items() if a.id() in tofinalize and a.category() in categories}  # precompute
        srcbox = {k:bb.clone().maxsquare().dilate(1.2) for (k,bb) in dstbox.items()}                            
        vo.activitymap(lambda a: a.confidence(0.1*a.confidence()) if (a.id() in tofinalize and
                                                                      a._label in categories and
                                                                      not any([(af._label == a._label and
                                                                                af._id != a._id and
                                                                                af._actorid != a._actorid and 
                                                                                srcbox[a._id].hasintersection(dstbox[af._id]))
                                                                               for af in vo.activities().values() if af.during_interval(a._startframe, a._endframe, inclusive=True)])) else a)
        
        # Activity group suppression:  Group activities may have at most one activity detection of this type per group in a spatial region surrounding the actor
        tosuppress = set(['hand_interacts_with_person', 'person_embraces_person', 'person_transfers_object', 'person_steals_object', 'person_purchases', 'person_talks_to_person'])
        suppressed = set([])
        activitybox = {a.id():vo.track(a.actorid()).boundingbox(a.startframe(), a.endframe()) for a in vo.activities().values() if a.id() in tofinalize and a.category() in tosuppress}
        activitybox = {k:bb.dilate(1.2).maxsquare() if bb is not None else bb for (k,bb) in activitybox.items()}
        candidates = [a for a in vo.activities().values() if a.id() in tofinalize]
        for a in sorted(candidates, key=lambda a: a.confidence(), reverse=True):  # decreasing confidence
            if a.category() in tosuppress:
                for o in candidates:  # other activities
                    if (o._actorid != a._actorid and  # different tracks
                        o._label == a._label and  # same category
                        o.confidence() <= a.confidence() and   # lower confidence
                        o._id not in suppressed and  # not already suppressed
                        o.during_interval(a.startframe(), a.endframe()) and # overlaps temporally by at least one frame
                        (activitybox[a._id] is not None and activitybox[o._id] is not None) and   # has valid tracks
                        activitybox[a._id].hasintersection(activitybox[o._id]) and  # has coarse overlap 
                        vo.track(a.actorid()).clone().maxsquare().dilate(1.2).segment_maxiou(vo.track(o.actorid()), a.startframe(), a.endframe()) > 0):  # has fine overlap "close by"
                        suppressed.add(o.id())  # greedy non-maximum suppression of lower confidence activity detection
        vo.activityfilter(lambda a: a.id() not in tofinalize or a.id() not in suppressed)

        # Activity duration
        vo.activitymap(lambda a: a.padto(5) if a.id() in tofinalize and a.category() in ['person_talks_to_person', 'person_interacts_with_laptop', 'person_reads_document', 'person_purchases'] else a)   
        vo.activitymap(lambda a: a.duration(2, centered=False) if a.id() in tofinalize and a.category() in ['person_opens_vehicle_door', 'person_closes_vehicle_door'] else a)
        vo.activitymap(lambda a: a.duration(2, centered=True) if a.id() in tofinalize and a.category() in ['person_enters_scene_through_structure', 'person_exits_scene_through_structure'] else a)
        vo.activitymap(lambda a: a.startframe(0) if a.id() in tofinalize and a.startframe() < 0 else a)

        # Activity confidence
        if activityconf is not None:
            vo.activityfilter(lambda a: a.id() not in tofinalize or a.confidence() >= activityconf)
    
        return vo

        
    def __call__(self, vi, activityiou=0.1, mirror=False, minprob=0.04, trackconf=0.2, maxdets=105, avgdets=70, throttle=True, buffered=True, finalized=True, mintracklen=None):
        (n,m,dt) = (self.temporal_support(), self.temporal_stride(), 1)  
        aa = self._allowable_activities  # dictionary mapping of allowable classified activities to output names        
        f_encode = self.totensor(training=False, validation=False, show=False, doflip=False)  # video -> tensor CxNxHxW
        f_mirror = lambda t: (t, torch.from_numpy(np.copy(np.flip(np.asarray(t), axis=3))))  # CxNxHxW -> CxNxHx(-W), np.flip is much faster than torch.flip, faster than encode mirror=True, np.flip returns a view which must be copied
        f_totensor = lambda v: (f_encode(v.clone(sharedarray=True) if mirror else v),) if (not mirror or v.actor().category() != 'person') else f_mirror(f_encode(v.clone(sharedarray=True)))  # do not mirror vehicle activities
        f_totensorlist = lambda V: [t for v in V for t in f_totensor(v)]        
        def f_reduce(T,V):
            j = sum([v.actor().category() == 'person' for v in V])  # person mirrored, vehicle not mirrored
            (tm, t) = torch.split(T, (2*j, len(T)-2*j), dim=0)  # assumes sorted order, person first, only person/vehicle
            return torch.cat((torch.mean(tm.view(-1, 2, tm.shape[1]), dim=1), t), dim=0) if j>0 else T  # mean over mirror augmentation

        try:
            with torch.no_grad():                                
                vp = next(vi)  # peek in generator to create clip
                vi = itertools.chain([vp], vi)  # unpeek
                sw = vipy.util.Stopwatch() if throttle else None  # real-time framerate estimate
                framerate = vp.framerate()
                for (k, (vo,vc)) in enumerate(zip(vi, vp.stream(buffered=buffered).clip(n, m, continuous=True, activities=False, delay=dt))):
                    videotracks = [] if vc is None else [vt for vt in vc.trackfilter(lambda t: len(t)>=4 and (t.category() == 'person' or (t.category() == 'vehicle' and vo.track(t.id()).ismoving(k-10*n+dt, k+dt)))).tracksplit()]  # vehicle moved recently?
                    if throttle:
                        videotracks.sort(key=lambda v: v.actor().confidence(last=1))  # in-place                                            
                        numdets = (maxdets if ((avgdets is None) or (sw.duration()<=60) or ((sw.duration()>60) and ((k/sw.duration())/vp.framerate())>0.8)) else
                                   (avgdets if ((k/sw.duration())/vp.framerate())>0.67 else int(avgdets//2)))   # real-time throttle schedule
                        videotracks = videotracks[-numdets:] if (numdets is not None and len(videotracks)>numdets) else videotracks   # select only the most confident for detection                
                    videotracks.sort(key=lambda v: v.actor().category())  # in-place, for grouping mirrored encoding: person<vehicle

                    if len(videotracks)>0 and (k+dt > n): 
                        logits = self.forward(torch.stack(f_totensorlist(videotracks))) # augmented logits in track index order, copy
                        logits = f_reduce(logits, videotracks) if mirror else logits  # reduced logits in track index order
                        (actorid, actorcategory) = ([t.actorid() for t in videotracks], [t.actor().category() for t in videotracks])
                        dets = [vipy.activity.Activity(category=aa[category], shortlabel=self._class_to_shortlabel[category], startframe=k-n+dt, endframe=k+dt, confidence=sm, framerate=framerate, actorid=actorid[j], attributes={'pip':category, 'logit':float(logit)})
                                for (j, category_sm_logit) in enumerate(self.softmax(logits))  # (classname, softmax, logit), unsorted
                                for (category, sm, logit) in category_sm_logit
                                if ((category in aa) and   # requested activities only
                                    (actorcategory[j] in self._verb_to_noun[category]) and   # noun matching with category renaming dictionary
                                    sm>=minprob)]   # minimum probability for new activity detection
                        vo.assign(k+dt, dets, activityiou=activityiou, activitymerge=False, activitynms=True)   # assign new activity detections by non-maximum suppression (merge happens at the end)
                        del logits, dets, videotracks  # torch garabage collection

                    if not isinstance(finalized, bool) and k > 0 and k%finalized == 0:
                        self.finalize(vo, trackconf=trackconf, startframe=k-finalized-5, endframe=k-5, mintracklen=mintracklen)  
                        
                    yield vo

        except Exception as e:                
            raise

        finally:
            if not (finalized is False):
                self.finalize(vo, trackconf=trackconf, mintracklen=mintracklen) if finalized == True else self.finalize(vo, trackconf=trackconf, startframe=(k//finalized)*finalized-4, endframe=k, mintracklen=mintracklen)


class ActivityTrackerCap(ActivityTracker, CAP):
    def __init__(self, stride=3, activities=None, gpus=None, batchsize=None, calibrated=False, modelfile=None, calibrated_constant=-1.5, unitnorm=False):
        ActivityTracker. __init__(self, stride=stride, activities=activities, gpus=gpus, batchsize=batchsize, mlbl=False, mlfl=True, modelfile=modelfile)
        CAP.__init__(self, modelfile=modelfile, deterministic=False, pretrained=None, mlbl=None, mlfl=True, calibrated_constant=calibrated_constant, calibrated=calibrated, unitnorm=unitnorm)
        # FIXME: there is an issue with multiple inheritance and multi-gpu with default parameters here (unitnorm, mlfl), requires hardcoding currently 


