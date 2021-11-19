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


class ActivityRecognition(object):
    def __init__(self, pretrained=True):
        self.net =  None
        self._class_to_index = {}
        self._num_frames = 0

    def class_to_index(self, c=None):
        return self._class_to_index if c is None else self._class_to_index[c]
    
    def index_to_class(self, index=None):
        d = {v:k for (k,v) in self.class_to_index().items()}
        return d if index is None else d[index]
    
    def classlist(self):
        return [k for (k,v) in sorted(list(self.class_to_index().items()), key=lambda x: x[0])]  # sorted in index order

    def num_classes(self):
        return len(self.classlist())

    def fromindex(self, k):
        index_to_class = {v:k for (k,v) in self.class_to_index().items()}
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
    
    def __init__(self, pretrained=True, deterministic=False, modelfile=None, mlbl=False, mlfl=False, bce=False):

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
        self._bce = bce
        
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

        (loss, n_valid) = (0, 0)
        C = torch.tensor([self._class_to_weight[k] for (k,v) in sorted(self._class_to_index.items(), key=lambda x: x[1])], device=y_hat.device)  # inverse class frequency        
        for (yh, yhs, labelstr) in zip(y_hat, y_hat_softmax, Y):
            labels = json.loads(labelstr)
            if labels is None:
                continue  # skip me
            lbllist = [l for lbl in labels for l in lbl]  # list of multi-labels within clip (unpack from JSON to use default collate_fn)
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
                    # Pick all labels normalized (https://papers.nips.cc/paper/2019/file/da647c549dde572c2c5edc4f5bef039c-Paper.pdf
                    loss += float(w)*F.cross_entropy(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)

                if self._bce:
                    # Binary cross entropy for per-class calibration
                    loss += float(w)*F.binary_cross_entropy_with_logits(torch.unsqueeze(yh, dim=0), torch.tensor([self._class_to_index[y]], device=y_hat.device), weight=C)                    
                
            n_valid += 1
        loss = loss / float(max(1, n_valid))  # batch reduction: mean

        if logging:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

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
        self.net = heyvi.model.ResNets_3D_PyTorch.resnet.generate_model(50, n_classes=self.num_classes())
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
        
        net = heyvi.model.ResNets_3D_PyTorch.resnet.generate_model(50, n_classes=1139)
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
            v = v.download() if (not v.hasfilename() and v.hasurl()) else v  # fetch it if necessary, but do not do this during training!        
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

    def __init__(self, pretrained=True, deterministic=False, modelfile=None, mlbl=False, mlfl=False, bce=False):
        pl.LightningModule.__init__(self)
        ActivityRecognition.__init__(self)  

        self._input_size = 112
        self._num_frames = 16        
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._mlfl = mlfl
        self._mlbl = mlbl
        self._bce = bce        
        if deterministic:
            np.random.seed(42)
        
        # Generated using vipy.dataset.Dataset(...).multilabel_inverse_frequency_weight()
        self._class_to_weight = {'car_drops_off_person': 0.7858124882763793, 'car_moves': 0.18439798528529147, 'car_picks_up_person': 0.7380666753394193, 'car_reverses': 0.5753369570213479, 'car_starts': 0.47486292483745757, 'car_stops': 0.44244800737774037, 'car_turns_left': 0.7697107319736983, 'car_turns_right': 0.5412936796835607, 'hand_interacts_with_person': 0.2794031245117859, 'person_abandons_package': 1.0789960714517162, 'person_carries_heavy_object': 0.5032333530901552, 'person_closes_car_door': 0.46460114438995603, 'person_closes_car_trunk': 0.6824201392305784, 'person_closes_facility_door': 0.38990434394080076, 'person_embraces_person': 0.6457437695527715, 'person_enters_car': 0.6934926810021877, 'person_enters_scene_through_structure': 0.2586965095740063, 'person_exits_car': 0.6766386632434479, 'person_exits_scene_through_structure': 0.33054895987676847, 'person_interacts_with_laptop': 0.6720176496986436, 'person_loads_car': 0.6880555743488312, 'person_opens_car_door': 0.4069868136393968, 'person_opens_car_trunk': 0.6911966903970317, 'person_opens_facility_door': 0.3018924474724252, 'person_picks_up_object': 0.4298381074082487, 'person_purchases_from_cashier': 5.479834409621331, 'person_purchases_from_machine': 5.31528236654537, 'person_puts_down_object': 0.2804690906037155, 'person_reads_document': 0.5476186269530937, 'person_rides_bicycle': 1.6090962879286763, 'person_sits_down': 0.4750148103149501, 'person_stands_up': 0.5022364750834624, 'person_steals_object': 0.910991409921711, 'person_talks_on_phone': 0.15771902851484076, 'person_talks_to_person': 0.21362675034201736, 'person_texts_on_phone': 0.3328378404741194, 'person_transfers_object_to_car': 2.964890512157848, 'person_transfers_object_to_person': 0.6481292773603928, 'person_unloads_car': 0.515379337544623, 'person_walks': 6.341278284010202}
        
        # Generated using vipy.dataset.Dataset(...).class_to_index()
        self._class_to_index = {'car_drops_off_person': 0, 'car_moves': 1, 'car_picks_up_person': 2, 'car_reverses': 3, 'car_starts': 4, 'car_stops': 5, 'car_turns_left': 6, 'car_turns_right': 7, 'hand_interacts_with_person': 8, 'person_abandons_package': 9, 'person_carries_heavy_object': 10, 'person_closes_car_door': 11, 'person_closes_car_trunk': 12, 'person_closes_facility_door': 13, 'person_embraces_person': 14, 'person_enters_car': 15, 'person_enters_scene_through_structure': 16, 'person_exits_car': 17, 'person_exits_scene_through_structure': 18, 'person_interacts_with_laptop': 19, 'person_loads_car': 20, 'person_opens_car_door': 21, 'person_opens_car_trunk': 22, 'person_opens_facility_door': 23, 'person_picks_up_object': 24, 'person_purchases_from_cashier': 25, 'person_purchases_from_machine': 26, 'person_puts_down_object': 27, 'person_reads_document': 28, 'person_rides_bicycle': 29, 'person_sits_down': 30, 'person_stands_up': 31, 'person_steals_object': 32, 'person_talks_on_phone': 33, 'person_talks_to_person': 34, 'person_texts_on_phone': 35, 'person_transfers_object_to_car': 36, 'person_transfers_object_to_person': 37, 'person_unloads_car': 38, 'person_walks': 39}
        
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
            v = v.download() if (v.hasurl() and not v.hasfilename()) else v  # fetch it if necessary, but do not do this during training!        
            vc = v.clone()
            if training or validation:
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
                # - 
                
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

        raise  # test that there are no copies needed
        
        assert v is None or isinstance(v, vipy.video.Scene), "Invalid input"
        f = (lambda v, num_frames=self._num_frames, input_size=self._input_size, mean=self._mean, std=self._std, training=training, validation=validation, show=show, classname=self.__class__.__name__:
             PIP_370k._totensor(v, training, validation, input_size, num_frames, mean, std, noflip=['car_turns_left', 'car_turns_right', 'vehicle_turns_left', 'vehicle_turns_right', 'motorcycle_turns_left', 'motorcycle_turns_right'], show=show, doflip=doflip, asjson=asjson, classname=classname))
        return f(v) if v is not None else f


class CAP(PIP_370k, pl.LightningModule, ActivityRecognition):
    def __init__(self, modelfile=None, deterministic=False, pretrained=None, mlbl=None, mlfl=True, bce=False):
        pl.LightningModule.__init__(self)
        ActivityRecognition.__init__(self)  

        self._input_size = 112
        self._num_frames = 16        
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._mlfl = True
        self._mlbl = False
        self._bce = bce
        
        if deterministic:
            np.random.seed(42)
        
        # Generated using vipy.dataset.Dataset.multilabel_inverse_frequency_weight()
        # WARNING: this was truncated to one
        self._class_to_weight = {'person_talks_on_phone': 0.0071500571732126764, 'car_moves': 0.008414980303528122, 'person_talks_to_person': 0.009391174775630284, 'person_opens_facility_door': 0.011485169340670632, 'person_enters_scene_through_structure': 0.011577007698961155, 'hand_interacts_with_person': 0.012750528623581633, 'person_puts_down_object': 0.012799173860425943, 'person_stands_up': 0.012843569626160192, 'person_exits_scene_through_structure': 0.014980961043628886, 'person_texts_on_phone': 0.01499085586892757, 'person_closes_facility_door': 0.016113488440289907, 'person_sits_down': 0.017908321006202585, 'person_opens_car_door': 0.01857279522480972, 'person_stands_up_from_floor': 0.019361727910618066, 'person_picks_up_object': 0.019615611319993825, 'car_stops': 0.02019106261740665, 'person_closes_car_door': 0.021202018411369025, 'car_starts': 0.021670313551423053, 'person_carries_heavy_object': 0.02274928459304244, 'person_unloads_car': 0.02351927526525588, 'car_turns_right': 0.024701873211435525, 'person_reads_document': 0.02499051143387395, 'car_reverses': 0.026255434156376472, 'person_transfers_object_to_person': 0.026580220081877853, 'person_picks_up_object_from_floor': 0.029207808500200837, 'person_embraces_person': 0.02945420882971027, 'person_interacts_with_laptop': 0.029911212318566305, 'person_exits_car': 0.030878325568415993, 'person_closes_car_trunk': 0.03114216254299982, 'person_loads_car': 0.03139933495976206, 'person_opens_car_trunk': 0.03154267942003807, 'person_enters_car': 0.03164745668042723, 'car_picks_up_person': 0.033681585653241426, 'car_turns_left': 0.035125658444433834, 'car_drops_off_person': 0.03586046019364838, 'person_steals_object': 0.0415729855145331, 'person_picks_up_object_from_table': 0.04418573296650998, 'person_drops_object': 0.048065094048302635, 'person_abandons_package': 0.04923985842254567, 'person_jumps': 0.05345098137582828, 'person_takes_phone_from_pocket': 0.05462388692326024, 'person_trips_on_object_on_floor': 0.05498089026389937, 'person_puts_phone_into_pocket': 0.056183850529544055, 'person_drinks_from_mug': 0.05666046130633482, 'person_drinks_from_bottle': 0.057329881619530565, 'person_sweeps_floor': 0.06141749219518637, 'person_uses_television_remote': 0.06148643566096271, 'person_wipes_mouth_with_napkin': 0.0616838064677993, 'person_sprays_from_bottle': 0.06264264051441124, 'person_salutes': 0.0628090316681978, 'person_puts_down_object_on_table': 0.06571369688838084, 'person_looks_at_wristwatch': 0.06672561082585902, 'person_puts_hands_in_back_pockets': 0.06787709488015589, 'person_punches': 0.06849983752132058, 'person_drinks_from_straw': 0.06898414063150268, 'person_strokes_chin': 0.06910772776183577, 'person_exercises_with_jumping_jacks': 0.06940183386381935, 'person_opens_laptop': 0.07058098348291678, 'person_grabs_person_by_bicep': 0.07060789551353987, 'person_grabs_person_by_forearm': 0.07071022579689283, 'person_strokes_hair': 0.0711640253790199, 'person_grabs_person_by_hair': 0.07126797516777103, 'person_closes_laptop': 0.07141913528726114, 'person_cleans_table_with_rag': 0.07177957626166052, 'person_searches_in_bag': 0.07240861269366403, 'person_exercises_with_pushup': 0.07245544215462119, 'person_picks_up_object_from_bed': 0.07261587289604508, 'person_rides_bicycle': 0.07343091926113407, 'person_brushes_teeth': 0.07613928807717861, 'person_gestures_raise_hand': 0.07666348171707751, 'person_sits_crisscross': 0.0777406880176164, 'person_gestures_be_quiet': 0.0778337212322166, 'person_gestures_cut': 0.07796334578685198, 'person_shoves_person': 0.07798921962892592, 'person_squats': 0.07838299609818057, 'person_puts_on_hat': 0.07850485876811529, 'person_crosses_arms': 0.07882076865889508, 'person_lies_down_on_bed': 0.0791809589015662, 'person_takes_off_hat': 0.07954505147236295, 'person_waves_at_person': 0.08075149917222121, 'person_stirs_mug': 0.08080499469999347, 'person_gestures_behind_me': 0.08104660431869777, 'person_exercises_with_situp': 0.08108438212368972, 'person_points_at_person': 0.0813710070044297, 'person_claps_hands': 0.0813710070044297, 'person_pats_head': 0.0825550859557632, 'person_gestures_listen_closely': 0.08286354585573379, 'person_bumps_into_wall': 0.0831459710290662, 'person_gestures_come_here': 0.08328790679388197, 'person_eats_snack_from_bag': 0.08403922538881889, 'person_puts_on_glasses': 0.08409178869330584, 'person_gestures_blow_kiss': 0.08417884023439623, 'person_picks_up_object_from_countertop': 0.08545575718410064, 'person_gestures_stop': 0.08591091430900676, 'person_takes_off_glasses': 0.08607984438930144, 'person_takes_off_facemask': 0.08657028030837606, 'person_gestures_lower_hand': 0.08671760324527736, 'person_puts_on_facemask': 0.0874573927099682, 'person_pours_into_bowl': 0.08757414051616512, 'person_gestures_watch_closely': 0.08822794900516465, 'person_gestures_number_five': 0.08909798356438287, 'person_gestures_number_three': 0.08913053671877248, 'person_gestures_number_one': 0.08932635627948746, 'person_gestures_arms_x': 0.08935907655651291, 'person_gestures_thumbs_up': 0.08974733507235413, 'person_gestures_number_four': 0.08981969035319597, 'person_searches_in_box': 0.09051958404425983, 'person_karate_kicks': 0.09061798903721408, 'person_gestures_heart': 0.09078908783002615, 'person_gestures_call_me': 0.09171063120273695, 'person_karate_chop': 0.09211588142846427, 'person_covers_face_with_hands': 0.09212623829277955, 'person_drinks_from_cup': 0.09237747667762447, 'person_bounces_ball_on_floor': 0.09314634555146249, 'person_takes_off_headphones': 0.09357509742972007, 'person_puts_on_headphones': 0.09422567748137516, 'person_puts_down_object_on_countertop': 0.09437718336136768, 'person_bows': 0.0948116125142947, 'person_searches_in_couch': 0.09669055846186297, 'person_gestures_thumbs_down': 0.09682838430256516, 'person_exercises_with_lunges': 0.09696408981583088, 'person_picks_up_object_from_couch': 0.09775818840195175, 'person_hugs_stuffed_animal': 0.09944976722351416, 'person_reads_book': 0.09957154244868582, 'person_sneezees into arm': 0.10051239314918431, 'person_takes_object_from_bag': 0.10070841311819703, 'person_puts_down_object_on_couch': 0.10077857312928755, 'person_sneezes_into_hand': 0.10219953037255143, 'person_lies_down_on_floor': 0.10227736307765319, 'person_tucks_in_shirt': 0.10246240170736202, 'person_puts_feet_up': 0.10246595377450825, 'person_yawns': 0.10284787750290855, 'person_crosses_legs': 0.10306306675085773, 'person_laughs_with_person': 0.10310178497269419, 'person_takes_object_from_cabinet': 0.10344711927898777, 'person_clips_fingernails': 0.10358822887442899, 'person_closes_cabinet': 0.10364868957787093, 'person_opens_refrigerator': 0.10400807518189813, 'person_puts_object_in_cabinet': 0.10437215722553372, 'person_applies_deodorant': 0.10438608429579814, 'person_reads_magazine': 0.10452025664065136, 'person_lies_down_on_couch': 0.10479046066920758, 'person_rubs_neck': 0.10515098232727597, 'person_dances_in_place': 0.10528096409904394, 'person_looks_at_hands_in_lap': 0.10560618138496981, 'person_shrugs': 0.10615764969507409, 'person_opens_curtains': 0.1064289157366757, 'person_carries_groceries': 0.10677998502422664, 'person_closes_curtains': 0.10701196252783704, 'person_puts_on_scarf': 0.10707652735115192, 'person_holds_object_above_head': 0.10746708325959484, 'person_puts_object_into_bag': 0.10770431743897584, 'person_touches_earlobe': 0.10789486023851405, 'person_crawls': 0.10814051028495372, 'person_puts_down_object_on_bed': 0.10908517308037075, 'person_swats_bug': 0.10929671998175639, 'person_rubs_foot': 0.1093457099951951, 'person_rubs_eyes': 0.10944382189290276, 'person_taps_object_with_finger': 0.10959132030515735, 'person_twirls': 0.10993703424933766, 'person_files_fingernails': 0.10993703424933766, 'person_gestures_swipe_up': 0.11013924537090604, 'person_eats_with_hands': 0.11072316058030798, 'person_gestures_swipe_left': 0.11113247931061442, 'person_closes_refrigerator': 0.1111555983642474, 'person_scratches_face': 0.11180122777235575, 'person_gestures_peace': 0.11185248922479608, 'person_searches_under_couch': 0.11190379770609186, 'person_gestures_swipe_down': 0.11198350367806151, 'person_tears_paper': 0.11205800597118983, 'person_cracks_knuckles': 0.11216104781576104, 'person_folds_towel': 0.11236065501421866, 'person_takes_off_scarf': 0.11262669819417719, 'person_takes_medicine_pills': 0.11262709095072956, 'person_untucks_shirt': 0.11283546669716941, 'person_folds_pants': 0.11297390288995361, 'person_searches_in_kitchen_drawer': 0.11309702318000939, 'person_gestures_swipe_right': 0.1132382679012143, 'person_eats_with_utensil': 0.11360054487971481, 'person_waves_hand_over_object': 0.11367673765110917, 'person_squeezes_object': 0.11372973379919825, 'person_folds_shirt': 0.11375160594315399, 'person_licks_fingers': 0.11442320778577873, 'person_folds_socks': 0.11491150791364328, 'person_puts_down_person': 0.11519465416162661, 'person_grabs_person_by_shoulder': 0.11572593880421266, 'person_carries_furniture': 0.11583583998066489, 'person_carries_laundry_basket': 0.11628835960688708, 'person_opens_cabinet': 0.1163328202475424, 'person_crumples_paper': 0.11633298950847891, 'person_spins_person_around': 0.11644404725502637, 'person_picks_up_object_from_cabinet': 0.11722742863973104, 'person_brushes_hair': 0.11752581209517853, 'person_puts_fingers_in_ear': 0.11756639951772542, 'person_wiggles_hips': 0.11819296463143424, 'person_nudges_person_with_elbow': 0.11864035680381292, 'person_touches_face_of_person': 0.1199952183961044, 'person_touches_back_of_person': 0.1201040449053256, 'person_puts_down_object_into_cabinet': 0.12100708283694457, 'person_brushes_hair_of_person': 0.12197513949964013, 'person_searches_in_backback': 0.12213018201245313, 'person_blows_nose': 0.12264971292070401, 'person_pounds_shoulders_of_person': 0.12281303232588005, 'person_takes_off_shoes': 0.12350532456683978, 'person_jumps_on_couch': 0.12378164652675162, 'person_mops': 0.1252311493836141, 'person_snaps_fingers': 0.1256180633363956, 'person_puts_on_shoes': 0.12636147834728292, 'person_washes_hands': 0.12684469836917986, 'person_puts_on_sunglasses': 0.12747187073013944, 'person_flosses': 0.1277895646931798, 'person_kneels': 0.1278423621928503, 'person_reads_newspaper': 0.12825987329089392, 'person_kisses_cheek_of_person': 0.12886966666628646, 'person_puts_object_into_backpack': 0.12991164284856274, 'person_polishes_car_with_rag': 0.13003746215313447, 'person_unloads_clothes_from_suitcase': 0.1302110517855032, 'person_dries_hair_with_towel': 0.1302457442601603, 'person_dries_hands_with_towel': 0.1306844570815548, 'person_turns_off_faucet': 0.13197629371214256, 'person_turns_on_faucet': 0.13294031635069914, 'person_puts_on_jacket': 0.1332673096496579, 'person_searches_under_bed': 0.13337904811332982, 'person_dries_dish': 0.1335343175960496, 'person_transfers_object_to_car': 0.13530242872949544, 'person_drums_on_chest': 0.13582977672565716, 'person_stretches_back': 0.13635674842915196, 'person_puts_on_gloves': 0.13785597770489835, 'person_gestures_hang_loose': 0.1381371908263195, 'person_applies_lip_makeup': 0.13821545552367154, 'person_picks_up_object_from_shelf': 0.13901394290468203, 'person_takes_object_from_backpack': 0.1390887336531267, 'person_slaps_hands_on_thighs': 0.13932054768662494, 'person_takes_off_sunglasses': 0.13955965617807797, 'person_paints_fingernails': 0.141502482018144, 'person_opens_home_window': 0.14158460766063857, 'person_applies_eye_makeup': 0.14191406573547427, 'person_closes_home_window': 0.14205455016553906, 'person_climbs_on_chair': 0.14286416865055518, 'person_opens_dresser_drawer': 0.14392346843615356, 'person_opens_closet_door': 0.14409349025356186, 'person_discards_trash': 0.14619789236857958, 'person_puts_down_object_on_shelf': 0.14623046508025042, 'person_climbs_off_chair': 0.1466640130633881, 'person_pulls_out_chair': 0.1467564560629468, 'person_stretches_arms_over_head': 0.14774021316767652, 'person_takes_off_gloves': 0.14792482083182573, 'person_hikes_up_pants': 0.14811795931953872, 'person_closes_closet_door': 0.14838824756647218, 'person_puts_on_wristwatch': 0.14846758639707738, 'person_prays': 0.149479337622108, 'person_opens_suitcase': 0.14999956703824965, 'person_washes_dish': 0.15061335914443671, 'person_interacts_with_tablet': 0.15067960407614595, 'person_sticks_out_tongue': 0.15114639343202, 'person_puts_up_picture_frame': 0.15124009857363935, 'person_stretches_arms_to_side': 0.1517687347875614, 'person_ties_jacket_around_waist': 0.15205524466480033, 'person_closes_dresser_drawer': 0.15256427704770498, 'person_takes_object_from_kitchen_drawer': 0.15292174958247323, 'person_covers_with_blanket': 0.1532186939054849, 'person_applies_foundation_makeup': 0.15362108249324954, 'person_takes_down_picture_frame': 0.15371788216715832, 'person_takes_selfie': 0.1544966934764283, 'person_walks_around_car': 0.1545946001262866, 'person_jumps_on_bed': 0.15469263094437558, 'person_zips_up_jacket': 0.15477837356130825, 'person_closes_suitcase': 0.1548318044264797, 'person_sets_upright_glass': 0.15528343666408673, 'person_takes_off_jacket': 0.1557468846718362, 'person_puts_object_into_purse': 0.15597843925785185, 'person_searches_in_purse': 0.1564786908269918, 'person_puts_on_socks': 0.15697187998993098, 'person_unties_jacket_around_waist': 0.1575376799198842, 'person_takes_off_backpack': 0.15780303861823264, 'person_arranges_flowers_in_vase': 0.15789662071150826, 'person_opens_kitchen_drawer': 0.15796859678500738, 'person_unzips_jacket': 0.15821465973786641, 'person_puts_on_backpack': 0.1591174653417599, 'person_closes_kitchen_drawer': 0.16220003797595112, 'person_puts_on_necklace': 0.1626663379799898, 'person_puts_object_into_kitchen_drawer': 0.16363350184412317, 'person_takes_off_socks': 0.1637836332966157, 'person_hits_person_with_pillow': 0.16383497582221643, 'person_climbs_on_couch': 0.16385203176578586, 'person_throws_object_into_air': 0.16542943627126713, 'person_takes_object_from_purse': 0.16550222455853475, 'person_kicks_car_tires': 0.1662919420581324, 'person_loads_clothes_into_suitcase': 0.16665572681128546, 'person_takes_object_from_basket': 0.16704418245100291, 'person_takes_off_necklace': 0.16720375531136414, 'person_burns_hand': 0.16766342199263248, 'person_walks_tiptoe': 0.16766342199263248, 'person_applies_sunscreen': 0.1694099159717224, 'person_pushes_in_chair': 0.17093473022331301, 'person_nods_head': 0.1730143822689931, 'person_puts_on_ring': 0.17320954516305123, 'person_puts_object_into_basket': 0.17345866516751823, 'person_covers_friend_with_blanket': 0.17375375997099735, 'person_shakes_head': 0.1741258236968453, 'person_takes_off_wristwatch': 0.1751258284273369, 'person_lights_candle': 0.17639210339788883, 'person_picks_up_person': 0.1775475101887047, 'person_shades_eyes': 0.17898039545068253, 'person_uncovers_friend_with_blanket': 0.17990433554519192, 'person_closes_clothes_washer': 0.18070391036983724, 'person_blows_into_hands': 0.18097201706178062, 'person_opens_clothes_washer': 0.18097201706178062, 'person_puts_on_belt': 0.1826837651926084, 'person_bumps_into_table': 0.18273429138522868, 'person_climbs_off_couch': 0.18411341811266432, 'person_uncovers_with_blanket': 0.1846559266696029, 'person_eats_apple': 0.1855135201515439, 'person_opens_box': 0.18707843481539896, 'person_takes_off_ring': 0.18750982244372041, 'person_throws_object_to_ground': 0.1885241723332923, 'person_locks_door_with_keys': 0.1891087434102948, 'person_unlocks_door_with_keys': 0.18954955633199708, 'person_vacuums_carpet': 0.18954955633199708, 'person_opens_microwave': 0.18967163518640828, 'person_extinguishes_candle': 0.18984457509671618, 'person_applies_facial_moisturizer': 0.19043737626797835, 'person_eats_banana': 0.1929986384487977, 'person_stubs_toe': 0.19395487244437154, 'person_puts_on_shirt': 0.194933929667213, 'person_closes_box': 0.195472979967372, 'person_puts_down_object_on_floor': 0.19599558450709312, 'person_throws_object_on_table': 0.19898065171230037, 'person_opens_jar': 0.20028758538528754, 'person_drinks_from_beverage_can': 0.20218778029011394, 'person_closes_microwave': 0.20322926171591793, 'person_exercises_with_plank': 0.2032303156689395, 'person_takes_clothes_from_dresser': 0.2043134664985597, 'person_carries_bicycle': 0.20500023445317672, 'person_dusts_furniture': 0.20813094360488035, 'person_closes_jar': 0.20921979330984586, 'person_knocks_over_glass': 0.21030196465455195, 'person_takes_off_belt': 0.21048341587513394, 'person_closes_door_with_foot': 0.21066518048297084, 'person_puts_clothes_into_dresser': 0.21066518048297084, 'person_kicks_object_to_person': 0.21194637619398804, 'person_pours_coffee_into_mug': 0.2139914728063862, 'person_buttons_shirt': 0.21535941093384248, 'person_falls_from_chair': 0.21573745621114188, 'person_puts_on_hoodie': 0.21625599556698347, 'person_opens_beverage_can': 0.21703761476804292, 'person_puts_on_earrings': 0.21786207374692124, 'person_climbs_on_table': 0.22102665713111078, 'person_climbs_off_table': 0.22195290532393117, 'person_takes_off_earrings': 0.2233976913912823, 'person_waters_houseplant': 0.2236024555447115, 'person_opens_oven_door': 0.22407580298906604, 'person_wraps_box': 0.22567093339433883, 'person_loads_clotheswasher': 0.22927657800684234, 'person_closes_oven_door': 0.2313013332980745, 'person_closes_door_with_hip': 0.23233359904693357, 'person_unbuttons_shirt': 0.2332220640528492, 'person_pushes_wheeled_cart': 0.2375367857831356, 'person_pulls_wheeled_cart': 0.23963681630577627, 'person_dries_hair_with_hairdryer': 0.24010854232212622, 'person_takes_off_shirt': 0.24034510246234506, 'person_irons_clothes': 0.2405821291906117, 'person_bumps_into_person': 0.2419661565158503, 'person_purchases_from_machine': 0.24256228370916671, 'person_purchases_from_cashier': 0.2500715967813632, 'person_takes_off_hoodie': 0.2512361266727912, 'person_flips_up_car_wipers': 0.25625029306647085, 'person_throws_object_to_person': 0.2584219057195765, 'person_climbs_up_stairs': 0.25924578002048915, 'person_drinks_from_shotglass': 0.26007492430626894, 'person_puts_on_pants': 0.26180106773743944, 'person_flips_down_car_wipers': 0.26203037486496267, 'person_spreads_tablecloth': 0.26287745581818994, 'person_takes_clothes_from_closet': 0.26373003135057327, 'person_folds_tablecloth': 0.26458815509683326, 'person_pours_liquid_into_cup': 0.26716946140944486, 'person_takes_selfie_with_person': 0.26837214411361965, 'person_washes_window': 0.2688077926543526, 'person_closes_car_hood': 0.271357373747809, 'person_climbs_down_stairs': 0.2719624069111263, 'person_covers_with_bedsheets': 0.2733971404050831, 'person_opens_car_hood': 0.274101437077843, 'person_puts_on_necktie': 0.2741340156665674, 'person_tickles_person': 0.275961854071584, 'person_uncovers_with_bedsheets': 0.27733168911014544, 'person_catches_dropped_object': 0.2784711362375529, 'person_lifts_dummbells': 0.2788003188563203, 'person_climbs_up_ladder': 0.28008068771444344, 'person_sets_table': 0.2810487085245164, 'person_puts_object_into_microwave': 0.2823920373735829, 'person_folds_blanket': 0.28267703244412545, 'person_interacts_with_handheld_game': 0.2839933399293134, 'person_looks_over_shoulder': 0.28532196374184826, 'person_sets_upright_furniture': 0.28532196374184826, 'person_catches_object_from_person': 0.2870003282344473, 'person_knocks_over_furniture': 0.2870003282344473, 'person_walks': 0.2893834863573906, 'person_puts_hair_in_ponytail': 0.290763145410346, 'person_climbs_down_ladder': 0.290763145410346, 'person_takes_off_pants': 0.2939159987943136, 'person_puts_clothes_into_closet': 0.2953393208223732, 'person_searches_in_cabinet': 0.2956973078779155, 'person_throws_object_on_bed': 0.29677649513294435, 'person_washes_face': 0.2971379768566142, 'person_carries_person_over_shoulder': 0.2972615412890275, 'person_takes_object_from_microwave': 0.2981634008906294, 'person_opens_can_with_can_opener': 0.29969321744383326, 'person_unfolds_blanket': 0.30117318394972875, 'person_dries_face_with_towel': 0.3041774052360103, 'person_loads_groceries_into_refrigerator': 0.306085670011644, 'person_puts_on_apron': 0.3076296078175035, 'person_makes_bed': 0.3099749415492761, 'person_takes_off_hairtie': 0.3119568785157037, 'person_unloads_clotheswasher': 0.31356076992195403, 'person_takes_off_apron': 0.31518123901715794, 'person_turns_off_fan': 0.3159977707244563, 'person_turns_on_lamp': 0.31681854415490945, 'person_turns_on_fan': 0.31788266098916557, 'person_opens_sliding_door': 0.31805773011640187, 'person_closes_sliding_door': 0.32141011725860374, 'person_takes_off_necktie': 0.324401966754362, 'person_hugs_person_from_behind': 0.324401966754362, 'person_loads_groceries_into_cabinet': 0.3257013070751405, 'person_carries_person_on_back': 0.3283314656787083, 'person_braids_hair_of_person': 0.3323573283368941, 'person_turns_on_stovetop': 0.3346368710552541, 'person_falls_from_bed': 0.3350905862080302, 'person_uses_bodyweight_scale': 0.3426267963473037, 'person_turns_off_lamp': 0.3431086905756403, 'person_vapes': 0.3435919422525074, 'person_turns_off_stovetop': 0.3460287645379862, 'person_closes_gate': 0.34701319914549117, 'person_opens_gate': 0.3475075199419947, 'person_puts_object_into_refrigerator': 0.3530701457147036, 'person_puts_object_into_oven': 0.3572873745649558, 'person_unscrews_lid_from_bottle': 0.35748569200453895, 'person_applies_shaving_cream': 0.35927876141278386, 'person_takes_object_from_refrigerator': 0.3595562632398506, 'person_shaves_face': 0.3646491464862186, 'person_takes_object_from_oven': 0.36930257807780786, 'person_screws_lid_to_bottle': 0.3758864083193841, 'person_loads_dryer': 0.3947415517787706, 'person_falls_while_standing': 0.4050786163671186, 'person_inserts_trashbag_into_trashcan': 0.47599885750525034, 'person_removes_trashbag_from_trashcan': 0.47707269147228026, 'person_gestures_zoom_out': 0.5082297479151672, 'person_gestures_zoom_in': 0.5291763101936665, 'person_unloads_dryer': 0.5421117311095116, 'person_handstand': 0.566009928072576, 'person_pets_dog': 0.6402894461923366, 'person_plugs_into_electrical_socket': 0.648803933508724, 'person_somersaults': 0.648803933508724, 'person_unplugs_from_electrical_socket': 0.6629083668458703, 'person_carries_person_on_shoulders': 0.6891250819188708, 'person_adjusts_thermostat': 0.7370099063422364, 'person_hugs_dog': 0.7719945537951908, 'person_attaches_leash_to_dog': 0.786936383868646, 'person_opens_dishwasher': 0.7920463603872736, 'person_closes_dishwasher': 0.7946263159585677, 'person_pets_cat': 0.8297628537390485, 'person_searches_jewelry_box': 0.8354461609564392, 'person_hugs_cat': 0.847049579858612, 'person_opens_jewelry_box': 0.8559658912255448, 'person_unattaches_leash_from_dog': 0.8620151201387994, 'person_unloads_dishwasher': 0.8712509964260009, 'person_loads_dishwasher': 0.8743737598540511, 'person_closes_jewelry_box': 0.9205670905633218, 'person_puts_object_into_toaster': 0.9642303517758113, 'person_takes_object_from_toaster': 0.9797199959810451, 'person_puts_on_boots': 0.9957154244868583, 'person_takes_off_boots': 1, 'person_unloads_box_onto_floor': 1, 'person_spills_on_table': 1, 'person_cleans_dryer_lint_trap': 1, 'person_exits_pool': 1, 'person_unloads_box_onto_table': 1, 'person_spills_on_floor': 1, 'person_texts_on_phone_while_sitting': 1, 'person_closes_mailbox': 1, 'person_opens_mailbox': 1, 'person_puts_object_into_box': 1, 'person_takes_object_from_box': 1, 'person_takes_down_smoke_detector': 1, 'person_puts_up_smoke_detector': 1, 'person_trips_on_stair': 1, 'person_jumps_into_pool': 1, 'person_falls_into_pool': 1, 'person_pulls_wheeled_trashcan': 1, 'person_uncrates_dog': 1, 'person_crates_dog': 1, 'person_pushes_wheeled_trashcan': 1, 'person_leaves_scene_through_structure': 1, 'person_crawls_out_from_under_vehicle': 1, 'person_feeds_dog': 1, 'person_feeds_cat': 1, 'person_points_to_dog': 1, 'person_throws_object_to_dog': 1, 'person_shakes_hand': 1, 'person_cleans_eyeglasses': 1, 'person_holds_hand': 1, 'person_embraces_sitting_person': 1}

        
        # Generated using vipy.dataset.Dataset.class_to_index()
        self._class_to_index = {'car_drops_off_person': 0, 'car_moves': 1, 'car_picks_up_person': 2, 'car_reverses': 3, 'car_starts': 4, 'car_stops': 5, 'car_turns_left': 6, 'car_turns_right': 7, 'hand_interacts_with_person': 8, 'person_abandons_package': 9, 'person_adjusts_thermostat': 10, 'person_applies_deodorant': 11, 'person_applies_eye_makeup': 12, 'person_applies_facial_moisturizer': 13, 'person_applies_foundation_makeup': 14, 'person_applies_lip_makeup': 15, 'person_applies_shaving_cream': 16, 'person_applies_sunscreen': 17, 'person_arranges_flowers_in_vase': 18, 'person_attaches_leash_to_dog': 19, 'person_blows_into_hands': 20, 'person_blows_nose': 21, 'person_bounces_ball_on_floor': 22, 'person_bows': 23, 'person_braids_hair_of_person': 24, 'person_brushes_hair': 25, 'person_brushes_hair_of_person': 26, 'person_brushes_teeth': 27, 'person_bumps_into_person': 28, 'person_bumps_into_table': 29, 'person_bumps_into_wall': 30, 'person_burns_hand': 31, 'person_buttons_shirt': 32, 'person_carries_bicycle': 33, 'person_carries_furniture': 34, 'person_carries_groceries': 35, 'person_carries_heavy_object': 36, 'person_carries_laundry_basket': 37, 'person_carries_person_on_back': 38, 'person_carries_person_on_shoulders': 39, 'person_carries_person_over_shoulder': 40, 'person_catches_dropped_object': 41, 'person_catches_object_from_person': 42, 'person_claps_hands': 43, 'person_cleans_dryer_lint_trap': 44, 'person_cleans_eyeglasses': 45, 'person_cleans_table_with_rag': 46, 'person_climbs_down_ladder': 47, 'person_climbs_down_stairs': 48, 'person_climbs_off_chair': 49, 'person_climbs_off_couch': 50, 'person_climbs_off_table': 51, 'person_climbs_on_chair': 52, 'person_climbs_on_couch': 53, 'person_climbs_on_table': 54, 'person_climbs_up_ladder': 55, 'person_climbs_up_stairs': 56, 'person_clips_fingernails': 57, 'person_closes_box': 58, 'person_closes_cabinet': 59, 'person_closes_car_door': 60, 'person_closes_car_hood': 61, 'person_closes_car_trunk': 62, 'person_closes_closet_door': 63, 'person_closes_clothes_washer': 64, 'person_closes_curtains': 65, 'person_closes_dishwasher': 66, 'person_closes_door_with_foot': 67, 'person_closes_door_with_hip': 68, 'person_closes_dresser_drawer': 69, 'person_closes_facility_door': 70, 'person_closes_gate': 71, 'person_closes_home_window': 72, 'person_closes_jar': 73, 'person_closes_jewelry_box': 74, 'person_closes_kitchen_drawer': 75, 'person_closes_laptop': 76, 'person_closes_mailbox': 77, 'person_closes_microwave': 78, 'person_closes_oven_door': 79, 'person_closes_refrigerator': 80, 'person_closes_sliding_door': 81, 'person_closes_suitcase': 82, 'person_covers_face_with_hands': 83, 'person_covers_friend_with_blanket': 84, 'person_covers_with_bedsheets': 85, 'person_covers_with_blanket': 86, 'person_cracks_knuckles': 87, 'person_crates_dog': 88, 'person_crawls': 89, 'person_crawls_out_from_under_vehicle': 90, 'person_crosses_arms': 91, 'person_crosses_legs': 92, 'person_crumples_paper': 93, 'person_dances_in_place': 94, 'person_discards_trash': 95, 'person_dries_dish': 96, 'person_dries_face_with_towel': 97, 'person_dries_hair_with_hairdryer': 98, 'person_dries_hair_with_towel': 99, 'person_dries_hands_with_towel': 100, 'person_drinks_from_beverage_can': 101, 'person_drinks_from_bottle': 102, 'person_drinks_from_cup': 103, 'person_drinks_from_mug': 104, 'person_drinks_from_shotglass': 105, 'person_drinks_from_straw': 106, 'person_drops_object': 107, 'person_drums_on_chest': 108, 'person_dusts_furniture': 109, 'person_eats_apple': 110, 'person_eats_banana': 111, 'person_eats_snack_from_bag': 112, 'person_eats_with_hands': 113, 'person_eats_with_utensil': 114, 'person_embraces_person': 115, 'person_embraces_sitting_person': 116, 'person_enters_car': 117, 'person_enters_scene_through_structure': 118, 'person_exercises_with_jumping_jacks': 119, 'person_exercises_with_lunges': 120, 'person_exercises_with_plank': 121, 'person_exercises_with_pushup': 122, 'person_exercises_with_situp': 123, 'person_exits_car': 124, 'person_exits_pool': 125, 'person_exits_scene_through_structure': 126, 'person_extinguishes_candle': 127, 'person_falls_from_bed': 128, 'person_falls_from_chair': 129, 'person_falls_into_pool': 130, 'person_falls_while_standing': 131, 'person_feeds_cat': 132, 'person_feeds_dog': 133, 'person_files_fingernails': 134, 'person_flips_down_car_wipers': 135, 'person_flips_up_car_wipers': 136, 'person_flosses': 137, 'person_folds_blanket': 138, 'person_folds_pants': 139, 'person_folds_shirt': 140, 'person_folds_socks': 141, 'person_folds_tablecloth': 142, 'person_folds_towel': 143, 'person_gestures_arms_x': 144, 'person_gestures_be_quiet': 145, 'person_gestures_behind_me': 146, 'person_gestures_blow_kiss': 147, 'person_gestures_call_me': 148, 'person_gestures_come_here': 149, 'person_gestures_cut': 150, 'person_gestures_hang_loose': 151, 'person_gestures_heart': 152, 'person_gestures_listen_closely': 153, 'person_gestures_lower_hand': 154, 'person_gestures_number_five': 155, 'person_gestures_number_four': 156, 'person_gestures_number_one': 157, 'person_gestures_number_three': 158, 'person_gestures_peace': 159, 'person_gestures_raise_hand': 160, 'person_gestures_stop': 161, 'person_gestures_swipe_down': 162, 'person_gestures_swipe_left': 163, 'person_gestures_swipe_right': 164, 'person_gestures_swipe_up': 165, 'person_gestures_thumbs_down': 166, 'person_gestures_thumbs_up': 167, 'person_gestures_watch_closely': 168, 'person_gestures_zoom_in': 169, 'person_gestures_zoom_out': 170, 'person_grabs_person_by_bicep': 171, 'person_grabs_person_by_forearm': 172, 'person_grabs_person_by_hair': 173, 'person_grabs_person_by_shoulder': 174, 'person_handstand': 175, 'person_hikes_up_pants': 176, 'person_hits_person_with_pillow': 177, 'person_holds_hand': 178, 'person_holds_object_above_head': 179, 'person_hugs_cat': 180, 'person_hugs_dog': 181, 'person_hugs_person_from_behind': 182, 'person_hugs_stuffed_animal': 183, 'person_inserts_trashbag_into_trashcan': 184, 'person_interacts_with_handheld_game': 185, 'person_interacts_with_laptop': 186, 'person_interacts_with_tablet': 187, 'person_irons_clothes': 188, 'person_jumps': 189, 'person_jumps_into_pool': 190, 'person_jumps_on_bed': 191, 'person_jumps_on_couch': 192, 'person_karate_chop': 193, 'person_karate_kicks': 194, 'person_kicks_car_tires': 195, 'person_kicks_object_to_person': 196, 'person_kisses_cheek_of_person': 197, 'person_kneels': 198, 'person_knocks_over_furniture': 199, 'person_knocks_over_glass': 200, 'person_laughs_with_person': 201, 'person_leaves_scene_through_structure': 202, 'person_licks_fingers': 203, 'person_lies_down_on_bed': 204, 'person_lies_down_on_couch': 205, 'person_lies_down_on_floor': 206, 'person_lifts_dummbells': 207, 'person_lights_candle': 208, 'person_loads_car': 209, 'person_loads_clothes_into_suitcase': 210, 'person_loads_clotheswasher': 211, 'person_loads_dishwasher': 212, 'person_loads_dryer': 213, 'person_loads_groceries_into_cabinet': 214, 'person_loads_groceries_into_refrigerator': 215, 'person_locks_door_with_keys': 216, 'person_looks_at_hands_in_lap': 217, 'person_looks_at_wristwatch': 218, 'person_looks_over_shoulder': 219, 'person_makes_bed': 220, 'person_mops': 221, 'person_nods_head': 222, 'person_nudges_person_with_elbow': 223, 'person_opens_beverage_can': 224, 'person_opens_box': 225, 'person_opens_cabinet': 226, 'person_opens_can_with_can_opener': 227, 'person_opens_car_door': 228, 'person_opens_car_hood': 229, 'person_opens_car_trunk': 230, 'person_opens_closet_door': 231, 'person_opens_clothes_washer': 232, 'person_opens_curtains': 233, 'person_opens_dishwasher': 234, 'person_opens_dresser_drawer': 235, 'person_opens_facility_door': 236, 'person_opens_gate': 237, 'person_opens_home_window': 238, 'person_opens_jar': 239, 'person_opens_jewelry_box': 240, 'person_opens_kitchen_drawer': 241, 'person_opens_laptop': 242, 'person_opens_mailbox': 243, 'person_opens_microwave': 244, 'person_opens_oven_door': 245, 'person_opens_refrigerator': 246, 'person_opens_sliding_door': 247, 'person_opens_suitcase': 248, 'person_paints_fingernails': 249, 'person_pats_head': 250, 'person_pets_cat': 251, 'person_pets_dog': 252, 'person_picks_up_object': 253, 'person_picks_up_object_from_bed': 254, 'person_picks_up_object_from_cabinet': 255, 'person_picks_up_object_from_couch': 256, 'person_picks_up_object_from_countertop': 257, 'person_picks_up_object_from_floor': 258, 'person_picks_up_object_from_shelf': 259, 'person_picks_up_object_from_table': 260, 'person_picks_up_person': 261, 'person_plugs_into_electrical_socket': 262, 'person_points_at_person': 263, 'person_points_to_dog': 264, 'person_polishes_car_with_rag': 265, 'person_pounds_shoulders_of_person': 266, 'person_pours_coffee_into_mug': 267, 'person_pours_into_bowl': 268, 'person_pours_liquid_into_cup': 269, 'person_prays': 270, 'person_pulls_out_chair': 271, 'person_pulls_wheeled_cart': 272, 'person_pulls_wheeled_trashcan': 273, 'person_punches': 274, 'person_purchases_from_cashier': 275, 'person_purchases_from_machine': 276, 'person_pushes_in_chair': 277, 'person_pushes_wheeled_cart': 278, 'person_pushes_wheeled_trashcan': 279, 'person_puts_clothes_into_closet': 280, 'person_puts_clothes_into_dresser': 281, 'person_puts_down_object': 282, 'person_puts_down_object_into_cabinet': 283, 'person_puts_down_object_on_bed': 284, 'person_puts_down_object_on_couch': 285, 'person_puts_down_object_on_countertop': 286, 'person_puts_down_object_on_floor': 287, 'person_puts_down_object_on_shelf': 288, 'person_puts_down_object_on_table': 289, 'person_puts_down_person': 290, 'person_puts_feet_up': 291, 'person_puts_fingers_in_ear': 292, 'person_puts_hair_in_ponytail': 293, 'person_puts_hands_in_back_pockets': 294, 'person_puts_object_in_cabinet': 295, 'person_puts_object_into_backpack': 296, 'person_puts_object_into_bag': 297, 'person_puts_object_into_basket': 298, 'person_puts_object_into_box': 299, 'person_puts_object_into_kitchen_drawer': 300, 'person_puts_object_into_microwave': 301, 'person_puts_object_into_oven': 302, 'person_puts_object_into_purse': 303, 'person_puts_object_into_refrigerator': 304, 'person_puts_object_into_toaster': 305, 'person_puts_on_apron': 306, 'person_puts_on_backpack': 307, 'person_puts_on_belt': 308, 'person_puts_on_boots': 309, 'person_puts_on_earrings': 310, 'person_puts_on_facemask': 311, 'person_puts_on_glasses': 312, 'person_puts_on_gloves': 313, 'person_puts_on_hat': 314, 'person_puts_on_headphones': 315, 'person_puts_on_hoodie': 316, 'person_puts_on_jacket': 317, 'person_puts_on_necklace': 318, 'person_puts_on_necktie': 319, 'person_puts_on_pants': 320, 'person_puts_on_ring': 321, 'person_puts_on_scarf': 322, 'person_puts_on_shirt': 323, 'person_puts_on_shoes': 324, 'person_puts_on_socks': 325, 'person_puts_on_sunglasses': 326, 'person_puts_on_wristwatch': 327, 'person_puts_phone_into_pocket': 328, 'person_puts_up_picture_frame': 329, 'person_puts_up_smoke_detector': 330, 'person_reads_book': 331, 'person_reads_document': 332, 'person_reads_magazine': 333, 'person_reads_newspaper': 334, 'person_removes_trashbag_from_trashcan': 335, 'person_rides_bicycle': 336, 'person_rubs_eyes': 337, 'person_rubs_foot': 338, 'person_rubs_neck': 339, 'person_salutes': 340, 'person_scratches_face': 341, 'person_screws_lid_to_bottle': 342, 'person_searches_in_backback': 343, 'person_searches_in_bag': 344, 'person_searches_in_box': 345, 'person_searches_in_cabinet': 346, 'person_searches_in_couch': 347, 'person_searches_in_kitchen_drawer': 348, 'person_searches_in_purse': 349, 'person_searches_jewelry_box': 350, 'person_searches_under_bed': 351, 'person_searches_under_couch': 352, 'person_sets_table': 353, 'person_sets_upright_furniture': 354, 'person_sets_upright_glass': 355, 'person_shades_eyes': 356, 'person_shakes_hand': 357, 'person_shakes_head': 358, 'person_shaves_face': 359, 'person_shoves_person': 360, 'person_shrugs': 361, 'person_sits_crisscross': 362, 'person_sits_down': 363, 'person_slaps_hands_on_thighs': 364, 'person_snaps_fingers': 365, 'person_sneezees into arm': 366, 'person_sneezes_into_hand': 367, 'person_somersaults': 368, 'person_spills_on_floor': 369, 'person_spills_on_table': 370, 'person_spins_person_around': 371, 'person_sprays_from_bottle': 372, 'person_spreads_tablecloth': 373, 'person_squats': 374, 'person_squeezes_object': 375, 'person_stands_up': 376, 'person_stands_up_from_floor': 377, 'person_steals_object': 378, 'person_sticks_out_tongue': 379, 'person_stirs_mug': 380, 'person_stretches_arms_over_head': 381, 'person_stretches_arms_to_side': 382, 'person_stretches_back': 383, 'person_strokes_chin': 384, 'person_strokes_hair': 385, 'person_stubs_toe': 386, 'person_swats_bug': 387, 'person_sweeps_floor': 388, 'person_takes_clothes_from_closet': 389, 'person_takes_clothes_from_dresser': 390, 'person_takes_down_picture_frame': 391, 'person_takes_down_smoke_detector': 392, 'person_takes_medicine_pills': 393, 'person_takes_object_from_backpack': 394, 'person_takes_object_from_bag': 395, 'person_takes_object_from_basket': 396, 'person_takes_object_from_box': 397, 'person_takes_object_from_cabinet': 398, 'person_takes_object_from_kitchen_drawer': 399, 'person_takes_object_from_microwave': 400, 'person_takes_object_from_oven': 401, 'person_takes_object_from_purse': 402, 'person_takes_object_from_refrigerator': 403, 'person_takes_object_from_toaster': 404, 'person_takes_off_apron': 405, 'person_takes_off_backpack': 406, 'person_takes_off_belt': 407, 'person_takes_off_boots': 408, 'person_takes_off_earrings': 409, 'person_takes_off_facemask': 410, 'person_takes_off_glasses': 411, 'person_takes_off_gloves': 412, 'person_takes_off_hairtie': 413, 'person_takes_off_hat': 414, 'person_takes_off_headphones': 415, 'person_takes_off_hoodie': 416, 'person_takes_off_jacket': 417, 'person_takes_off_necklace': 418, 'person_takes_off_necktie': 419, 'person_takes_off_pants': 420, 'person_takes_off_ring': 421, 'person_takes_off_scarf': 422, 'person_takes_off_shirt': 423, 'person_takes_off_shoes': 424, 'person_takes_off_socks': 425, 'person_takes_off_sunglasses': 426, 'person_takes_off_wristwatch': 427, 'person_takes_phone_from_pocket': 428, 'person_takes_selfie': 429, 'person_takes_selfie_with_person': 430, 'person_talks_on_phone': 431, 'person_talks_to_person': 432, 'person_taps_object_with_finger': 433, 'person_tears_paper': 434, 'person_texts_on_phone': 435, 'person_texts_on_phone_while_sitting': 436, 'person_throws_object_into_air': 437, 'person_throws_object_on_bed': 438, 'person_throws_object_on_table': 439, 'person_throws_object_to_dog': 440, 'person_throws_object_to_ground': 441, 'person_throws_object_to_person': 442, 'person_tickles_person': 443, 'person_ties_jacket_around_waist': 444, 'person_touches_back_of_person': 445, 'person_touches_earlobe': 446, 'person_touches_face_of_person': 447, 'person_transfers_object_to_car': 448, 'person_transfers_object_to_person': 449, 'person_trips_on_object_on_floor': 450, 'person_trips_on_stair': 451, 'person_tucks_in_shirt': 452, 'person_turns_off_fan': 453, 'person_turns_off_faucet': 454, 'person_turns_off_lamp': 455, 'person_turns_off_stovetop': 456, 'person_turns_on_fan': 457, 'person_turns_on_faucet': 458, 'person_turns_on_lamp': 459, 'person_turns_on_stovetop': 460, 'person_twirls': 461, 'person_unattaches_leash_from_dog': 462, 'person_unbuttons_shirt': 463, 'person_uncovers_friend_with_blanket': 464, 'person_uncovers_with_bedsheets': 465, 'person_uncovers_with_blanket': 466, 'person_uncrates_dog': 467, 'person_unfolds_blanket': 468, 'person_unloads_box_onto_floor': 469, 'person_unloads_box_onto_table': 470, 'person_unloads_car': 471, 'person_unloads_clothes_from_suitcase': 472, 'person_unloads_clotheswasher': 473, 'person_unloads_dishwasher': 474, 'person_unloads_dryer': 475, 'person_unlocks_door_with_keys': 476, 'person_unplugs_from_electrical_socket': 477, 'person_unscrews_lid_from_bottle': 478, 'person_unties_jacket_around_waist': 479, 'person_untucks_shirt': 480, 'person_unzips_jacket': 481, 'person_uses_bodyweight_scale': 482, 'person_uses_television_remote': 483, 'person_vacuums_carpet': 484, 'person_vapes': 485, 'person_walks': 486, 'person_walks_around_car': 487, 'person_walks_tiptoe': 488, 'person_washes_dish': 489, 'person_washes_face': 490, 'person_washes_hands': 491, 'person_washes_window': 492, 'person_waters_houseplant': 493, 'person_waves_at_person': 494, 'person_waves_hand_over_object': 495, 'person_wiggles_hips': 496, 'person_wipes_mouth_with_napkin': 497, 'person_wraps_box': 498, 'person_yawns': 499, 'person_zips_up_jacket': 500}

        
        self._verb_to_noun = {k:set(['car','vehicle','motorcycle','bus','truck']) if (k.startswith('car') or k.startswith('motorcycle') or k.startswith('vehicle')) else set(['person']) for k in self.classlist()}

        # Generated using vipy.dataset.Dataset.class_to_shortlabel()        
        self._class_to_shortlabel =  {'person_exits_scene_through_structure': 'Leaves scene', 'person_stands_up': 'Stands', 'person_talks_on_phone': 'Talks on phone', 'person_talks_to_person': 'Talks to person', 'person_enters_scene_through_structure': 'Comes into scene', 'person_closes_car_door': 'Unloads', 'person_picks_up_object': 'Picks up', 'person_enters_car': 'Opens', 'person_opens_facility_door': 'Opens door', 'person_sits_down': 'Sits down', 'person_opens_car_trunk': 'Closing', 'person_exits_car': 'Opens', 'person_texts_on_phone': 'Texts on phone', 'person_closes_facility_door': 'Closes door', 'person_opens_car_door': 'Opens', 'person_puts_down_object': 'Puts down', 'hand_interacts_with_person': 'Hold hands', 'person_closes_car_trunk': 'Unloading', 'person_purchases_from_cashier': 'Purchases (cashier)', 'person_carries_heavy_object': 'Carries heavy object', 'person_abandons_package': 'Abandons', 'person_reads_document': 'Reads', 'person_rides_bicycle': 'Rides', 'person_transfers_object_to_person': 'Transfers', 'person_embraces_person': 'Embraces person', 'person_unloads_car': 'Unloads', 'person_loads_car': 'Loads', 'person_interacts_with_laptop': 'Interacts with laptop', 'car_turns_right': 'Turns right', 'car_turns_left': 'Turns left', 'car_stops': 'Stops', 'car_reverses': 'Reverses', 'car_starts': 'Starts', 'car_drops_off_person': 'Stops', 'car_picks_up_person': 'Stops', 'person_steals_object': 'Steals', 'person_walks': 'Walks', 'person_purchases_from_machine': 'Purchases (machine)', 'person_transfers_object_to_car': 'Transferring', 'car_moves': 'moves', 'person_scratches_face': 'Scratches', 'person_drops_object': 'Drops object', 'person_picks_up_object_from_floor': 'Picks up object from floor', 'person_interacts_with_handheld_game': 'Uses', 'person_picks_up_object_from_countertop': 'Picks up object from countertop', 'person_puts_down_object_on_countertop': 'Puts down object on countertop', 'person_gestures_raise_hand': 'Gestures raise hand', 'person_gestures_lower_hand': 'Gestures lower hand', 'person_unlocks_door_with_keys': 'Unlocks door', 'person_locks_door_with_keys': 'Locks door', 'person_gestures_come_here': 'Come here sign', 'person_gestures_peace': 'Peace sign', 'person_nudges_person_with_elbow': 'Nudges with elbow', 'person_touches_back_of_person': 'Touches back of person', 'person_touches_face_of_person': 'Touches face of person', 'person_picks_up_object_from_bed': 'Picks up object from bed', 'person_puts_down_object_on_bed': 'Puts down object on bed', 'person_gestures_thumbs_down': 'Gestures thumbs down', 'person_waves_hand_over_object': 'Waves over', 'person_opens_laptop': 'Opens laptop', 'person_closes_laptop': 'Closes laptop', 'person_bumps_into_wall': 'Bumps into', 'person_waves_at_person': 'Waves at person', 'person_opens_car_hood': 'Opens', 'person_closes_car_hood': 'Closes', 'person_exercises_with_lunges': 'Exercises with lunges', 'person_reads_newspaper': 'Reads', 'person_points_at_person': 'Points to', 'person_gestures_thumbs_up': 'Gestures thumbs up', 'person_squeezes_object': 'Squeezes', 'person_cracks_knuckles': 'Cracks knuckles', 'person_closes_door_with_foot': 'Closes', 'person_gestures_swipe_left': 'Gestures swipe left', 'person_gestures_swipe_right': 'Gestures swipe right', 'person_opens_gate': 'Opens', 'person_closes_gate': 'Closes', 'person_trips_on_object_on_floor': 'Trips', 'person_stands_up_from_floor': 'Stands up from floor', 'person_carries_laundry_basket': 'Carries laundry basket', 'person_squats': 'Squats', 'person_picks_up_object_from_couch': 'Picks up object from couch', 'person_puts_down_object_on_couch': 'Puts down object on couch', 'person_picks_up_object_from_cabinet': 'Picks up', 'person_puts_down_object_into_cabinet': 'Puts down', 'person_exercises_with_jumping_jacks': 'Exercises with jumping jacks', 'person_reads_magazine': 'Reads', 'person_closes_door_with_hip': 'Closes', 'person_carries_groceries': 'Carries groceries', 'person_looks_at_hands_in_lap': 'Looks at', 'person_interacts_with_tablet': 'Uses', 'person_polishes_car_with_rag': 'Polishes', 'person_kisses_cheek_of_person': 'Kisses cheek of person', 'person_exercises_with_situp': 'Exercises with situp', 'person_carries_furniture': 'Carries', 'person_carries_bicycle': 'Carries', 'person_exercises_with_pushup': 'Exercises with pushup', 'person_bows': 'Bows', 'person_sits_crisscross': 'Sits', 'person_reads_book': 'Reads book', 'person_gestures_swipe_up': 'Gestures swipe up', 'person_gestures_swipe_down': 'Gestures swipe down', 'person_looks_at_wristwatch': 'Looks at wristwatch', 'person_kicks_car_tires': 'Kicks', 'person_taps_object_with_finger': 'Taps', 'person_pounds_shoulders_of_person': 'Pounds shoulders of person', 'person_pushes_wheeled_cart': 'Pushes', 'person_pulls_wheeled_cart': 'Pulls', 'person_flips_up_car_wipers': 'Flips up', 'person_flips_down_car_wipers': 'Flips down', 'person_walks_around_car': 'Walks around', 'person_opens_sliding_door': 'Opens', 'person_closes_sliding_door': 'Closes', 'person_drinks_from_bottle': 'Drinks from bottle', 'person_shoves_person': 'Shoves', 'person_jumps': 'Jumps', 'person_eats_snack_from_bag': 'Eats snack from bag', 'person_takes_off_glasses': 'Takes off glasses', 'person_puts_on_glasses': 'Puts on glasses', 'person_lies_down_on_couch': 'Lies down', 'person_lies_down_on_bed': 'Lies down on bed', 'person_puts_on_hat': 'Puts on hat', 'person_takes_off_hat': 'Takes off hat', 'person_puts_on_shoes': 'Puts on shoes', 'person_takes_off_shoes': 'Takes off shoes', 'person_laughs_with_person': 'Laughs', 'person_lies_down_on_floor': 'Lies down', 'person_puts_on_scarf': 'Puts on scarf', 'person_takes_off_scarf': 'Takes off scarf', 'person_takes_off_headphones': 'Takes off headphones', 'person_puts_on_headphones': 'Puts on headphones', 'person_takes_phone_from_pocket': 'Takes phone from pocket', 'person_puts_phone_into_pocket': 'Puts phone in', 'person_puts_on_gloves': 'Puts on gloves', 'person_takes_off_gloves': 'Takes off gloves', 'person_wipes_mouth_with_napkin': 'Wipes mouth with napkin', 'person_drinks_from_straw': 'Drinks', 'person_grabs_person_by_forearm': 'Grabs', 'person_puts_on_jacket': 'Puts on jacket', 'person_zips_up_jacket': 'Zips up jacket', 'person_unzips_jacket': 'Unzips jacket', 'person_takes_off_jacket': 'Takes off jacket', 'person_grabs_person_by_bicep': 'Grabs', 'person_grabs_person_by_shoulder': 'Grabs', 'person_spins_person_around': 'Spins around', 'person_grabs_person_by_hair': 'Grabs', 'person_carries_person_over_shoulder': 'Carries', 'person_puts_down_person': 'Puts down', 'person_picks_up_person': 'Picks up', 'person_searches_in_backback': 'Searches in backback', 'person_takes_object_from_backpack': 'Takes object from backpack', 'person_puts_object_into_backpack': 'Puts object into backpack', 'person_searches_in_purse': 'Searches', 'person_puts_object_into_purse': 'Puts object into purse', 'person_takes_object_from_purse': 'Takes from', 'person_takes_off_facemask': 'Takes off facemask', 'person_puts_on_facemask': 'Puts on facemask', 'person_sweeps_floor': 'Sweeps', 'person_vacuums_carpet': 'Vacuums carpet', 'person_tucks_in_shirt': 'Tucks in shirt', 'person_untucks_shirt': 'Untucks', 'person_applies_deodorant': 'Uses deodorant', 'person_claps_hands': 'Claps', 'person_puts_feet_up': 'Puts feet up', 'person_climbs_up_stairs': 'Climbs up stairs', 'person_climbs_down_stairs': 'Climbs down stairs', 'person_kneels': 'Kneels', 'person_jumps_into_pool': 'Jumps into pool', 'person_exits_pool': 'Exits pool', 'person_puts_on_belt': 'Puts on belt', 'person_takes_off_belt': 'Takes off', 'person_picks_up_object_from_table': 'Picks up object from table', 'person_drinks_from_shotglass': 'Drinks from shotglass', 'person_puts_down_object_on_table': 'Puts down object on table', 'person_karate_kicks': 'Karate kicks', 'person_karate_chop': 'Karate chop', 'person_hikes_up_pants': 'Adjusts pants', 'person_opens_curtains': 'Opens curtains', 'person_closes_curtains': 'Closes curtains', 'person_puts_on_necklace': 'Puts on necklace', 'person_takes_off_necklace': 'Takes off', 'person_slaps_hands_on_thighs': 'Slaps', 'person_flosses': 'Flosses', 'person_plugs_into_electrical_socket': 'Plugs', 'person_unplugs_from_electrical_socket': 'Unplugs', 'person_brushes_teeth': 'Brushes', 'person_puts_on_socks': 'Puts on socks', 'person_takes_off_socks': 'Takes off socks', 'person_puts_on_sunglasses': 'Puts on sunglasses', 'person_takes_off_sunglasses': 'Takes off', 'person_punches': 'Punches', 'person_brushes_hair': 'Brushes hair', 'person_crawls': 'Crawls', 'person_applies_sunscreen': 'Uses sunscreen', 'person_drinks_from_mug': 'Drinks from mug', 'person_stretches_back': 'Stretches back', 'person_stretches_arms_to_side': 'Stretches arms to side', 'person_stretches_arms_over_head': 'Stretches arms over head', 'person_unscrews_lid_from_bottle': 'Unscrews lid from bottle', 'person_pours_liquid_into_cup': 'Pours liquid into cup', 'person_screws_lid_to_bottle': 'Screws lid to bottle', 'person_ties_jacket_around_waist': 'Ties', 'person_unties_jacket_around_waist': 'Unties', 'person_crosses_arms': 'Crosses arms', 'person_opens_home_window': 'Opens home window', 'person_closes_home_window': 'Closes home window', 'person_puts_on_wristwatch': 'Puts on wristwatch', 'person_takes_off_wristwatch': 'Takes off', 'person_turns_off_fan': 'Turns off fan', 'person_turns_on_fan': 'Turns on fan', 'person_puts_on_ring': 'Puts on ring', 'person_takes_off_ring': 'Takes off', 'person_opens_closet_door': 'Opens', 'person_closes_closet_door': 'Closes', 'person_pulls_wheeled_trashcan': 'Pulls trashcan', 'person_pushes_wheeled_trashcan': 'Pushes trashcan', 'person_drums_on_chest': 'Drums', 'person_eats_with_hands': 'Eats with hands', 'person_snaps_fingers': 'Snaps', 'person_folds_socks': 'Folds socks', 'person_dries_dish': 'Dries dish', 'person_puts_on_earrings': 'Puts on earrings', 'person_takes_off_earrings': 'Takes off', 'person_folds_pants': 'Folds pants', 'person_sprays_from_bottle': 'Sprays from bottle', 'person_opens_microwave': 'Opens microwave', 'person_puts_object_into_microwave': 'Puts object into microwave', 'person_closes_microwave': 'Closes microwave', 'person_takes_object_from_microwave': 'Takes object from microwave', 'person_unloads_dishwasher': 'Unloads dishwasher', 'person_washes_hands': 'Washes hands', 'person_dries_hands_with_towel': 'Dries hands with towel', 'person_washes_dish': 'Washes dish', 'person_applies_lip_makeup': 'Applies', 'person_opens_refrigerator': 'Opens refrigerator', 'person_takes_object_from_refrigerator': 'Takes object from refrigerator', 'person_closes_refrigerator': 'Closes refrigerator', 'person_puts_object_into_refrigerator': 'Puts object into refrigerator', 'person_eats_with_utensil': 'Eats with utensil', 'person_opens_cabinet': 'Opens cabinet', 'person_closes_cabinet': 'Closes cabinet', 'person_licks_fingers': 'Licks', 'person_applies_facial_moisturizer': 'Uses moisturizer', 'person_unloads_clotheswasher': 'Unloads', 'person_loads_dryer': 'Loads', 'person_takes_clothes_from_dresser': 'Takes from', 'person_puts_clothes_into_dresser': 'Puts into', 'person_removes_trashbag_from_trashcan': 'Removes trashbag from trashcan', 'person_inserts_trashbag_into_trashcan': 'Inserts trashbag into trashcan', 'person_turns_on_lamp': 'Turns on', 'person_turns_off_lamp': 'Turns off', 'person_throws_object_to_person': 'Throws object to person', 'person_catches_object_from_person': 'Catches', 'person_waters_houseplant': 'Waters', 'person_applies_foundation_makeup': 'Applies', 'person_folds_shirt': 'Folds shirt', 'person_takes_medicine_pills': 'Takes pills', 'person_drinks_from_cup': 'Drinks from cup', 'person_opens_mailbox': 'Opens', 'person_closes_mailbox': 'Closes mailbox', 'person_blows_into_hands': 'Blows', 'person_puts_fingers_in_ear': 'Put into ears', 'person_cleans_table_with_rag': 'Cleans table with rag', 'person_loads_dishwasher': 'Loads', 'person_discards_trash': 'Discards trash', 'person_lights_candle': 'Lights candle', 'person_extinguishes_candle': 'Extinguishes candle', 'person_falls_into_pool': 'Falls into pool', 'person_yawns': 'Yawns', 'person_opens_dresser_drawer': 'Opens', 'person_closes_dresser_drawer': 'Closes', 'person_pulls_out_chair': 'Pulls out chair', 'person_pushes_in_chair': 'Pushes in chair', 'person_kicks_object_to_person': 'Kicks object', 'person_puts_clothes_into_closet': 'Puts into', 'person_takes_clothes_from_closet': 'Takes clothes from closet', 'person_looks_over_shoulder': 'Looks over', 'person_pets_dog': 'Pets dog', 'person_nods_head': 'Nods head', 'person_shakes_head': 'Shakes head', 'person_applies_eye_makeup': 'Applies', 'person_loads_clotheswasher': 'Loads', 'person_unbuttons_shirt': 'Unbuttons', 'person_buttons_shirt': 'Buttons shirt', 'person_opens_jar': 'Opens', 'person_closes_jar': 'Closes', 'person_falls_from_bed': 'Falls from bed', 'person_falls_from_chair': 'Falls from chair', 'person_hugs_cat': 'Hugs cat', 'person_prays': 'Prays', 'person_takes_selfie': 'Takes selfie', 'person_puts_hair_in_ponytail': 'Puts hair in ponytail', 'person_takes_off_hairtie': 'Removes hairtie', 'person_hugs_dog': 'Hugs dog', 'person_pets_cat': 'Pets cat', 'person_picks_up_object_from_shelf': 'Picks up object from shelf', 'person_puts_down_object_on_shelf': 'Puts down object on shelf', 'person_shrugs': 'Shrugs', 'person_opens_can_with_can_opener': 'Opens', 'person_rubs_neck': 'Rubs', 'person_takes_selfie_with_person': 'Takes selfie', 'person_wiggles_hips': 'Wiggles', 'person_dances_in_place': 'Dances in place', 'person_trips_on_stair': 'Trips on stair', 'person_sneezees into arm': 'Sneezees into arm', 'person_throws_object_on_table': 'Throws', 'person_touches_earlobe': 'Touches ear', 'person_twirls': 'Twirls', 'person_unloads_dryer': 'Unloads', 'person_shades_eyes': 'Shades eyes', 'person_washes_face': 'Washes', 'person_dries_face_with_towel': 'Dries', 'person_throws_object_to_ground': 'Throws', 'person_tickles_person': 'Tickles', 'person_bumps_into_person': 'Bumps into person', 'person_throws_object_on_bed': 'Throws', 'person_rubs_eyes': 'Rubs', 'person_stubs_toe': 'Stubs toe', 'person_sneezes_into_hand': 'Sneezes into hand', 'person_vapes': 'Vapes', 'person_tears_paper': 'Tears paper', 'person_spills_on_table': 'Spills on table', 'person_sets_upright_glass': 'Sets upright glass', 'person_covers_friend_with_blanket': 'Covers friend with blanket', 'person_uncovers_friend_with_blanket': 'Uncovers friend with blanket', 'person_takes_off_boots': 'Takes off boots', 'person_puts_on_boots': 'Puts on boots', 'person_uses_television_remote': 'Uses television remote', 'person_sets_table': 'Sets table', 'person_puts_on_apron': 'Puts on apron', 'person_takes_off_apron': 'Takes off apron', 'person_turns_on_stovetop': 'Turns on stovetop', 'person_turns_off_stovetop': 'Turns off stovetop', 'person_spreads_tablecloth': 'Spreads tablecloth', 'person_folds_tablecloth': 'Folds tablecloth', 'person_folds_blanket': 'Folds blanket', 'person_unfolds_blanket': 'Unfolds blanket', 'person_covers_with_blanket': 'Covers with blanket', 'person_uncovers_with_blanket': 'Uncovers with blanket', 'person_crumples_paper': 'Crumples paper', 'person_sets_upright_furniture': 'Sets upright furniture', 'person_knocks_over_furniture': 'Knock over furniture', 'person_knocks_over_glass': 'Knocks over glass', 'person_loads_groceries_into_refrigerator': 'Loads groceries (refrigerator)', 'person_uncovers_with_bedsheets': 'Uncovers with bedsheets', 'person_covers_with_bedsheets': 'Covers with bedsheets', 'person_clips_fingernails': 'Clips fingernails', 'person_loads_groceries_into_cabinet': 'Loads groceries (cabinet)', 'person_files_fingernails': 'Files fingernails', 'person_loads_clothes_into_suitcase': 'Loads clothes into suitcase', 'person_mops': 'Mops', 'person_spills_on_floor': 'Spills on floor', 'person_washes_window': 'Washes window', 'person_unloads_clothes_from_suitcase': 'Unloads clothes from suitcase', 'person_puts_on_backpack': 'Puts on backpack', 'person_takes_off_backpack': 'Takes off backpack', 'person_opens_jewelry_box': 'Opens jewelry box', 'person_searches_jewelry_box': 'Searches jewelry box', 'person_closes_jewelry_box': 'Closes jewelry box', 'person_lifts_dummbells': 'Lifts dumbbells', 'person_swats_bug': 'Swats bug', 'person_irons_clothes': 'Irons clothes', 'person_uses_bodyweight_scale': 'Uses scale', 'person_makes_bed': 'Makes bed', 'person_unloads_box_onto_floor': 'Unloads box (floor)', 'person_unloads_box_onto_table': 'Unloads box (table)', 'person_searches_in_cabinet': 'Searches (cabinet)', 'person_dusts_furniture': 'Dusts furniture', 'person_handstand': 'Handstands', 'person_opens_beverage_can': 'Opens beverage can', 'person_drinks_from_beverage_can': 'Drinks from beverage can', 'person_somersaults': 'Somersaults', 'person_jumps_on_couch': 'Jumps on couch', 'person_climbs_up_ladder': 'Climbs up ladder', 'person_climbs_down_ladder': 'Climbs down ladder', 'person_covers_face_with_hands': 'Covers face', 'person_climbs_on_table': 'Climbs on table', 'person_climbs_off_table': 'Climbs off table', 'person_arranges_flowers_in_vase': 'Arranges flowers', 'person_crosses_legs': 'Crosses legs', 'person_climbs_on_chair': 'Climbs on chair', 'person_climbs_off_chair': 'Climbs off chair', 'person_bumps_into_table': 'Bumps into', 'person_climbs_on_couch': 'Climbs on couch', 'person_climbs_off_couch': 'Climbs off couch', 'person_opens_box': 'Opens box', 'person_closes_box': 'Closes box', 'person_cleans_dryer_lint_trap': 'Cleans', 'person_strokes_hair': 'Strokes hair', 'person_puts_on_shirt': 'Puts on shirt', 'person_takes_off_shirt': 'Takes off', 'person_applies_shaving_cream': 'Applies', 'person_shaves_face': 'Shaves', 'person_jumps_on_bed': 'Jumps on bed', 'person_puts_hands_in_back_pockets': 'Puts hands in', 'person_searches_in_bag': 'Searches in bag', 'person_folds_towel': 'Folds towel', 'person_puts_on_pants': 'Puts on pants', 'person_takes_object_from_toaster': 'Takes from', 'person_puts_object_into_toaster': 'Puts into', 'person_searches_under_couch': 'Searches', 'person_puts_up_picture_frame': 'Puts up picture frame', 'person_takes_down_picture_frame': 'Takes down picture frame', 'person_strokes_chin': 'Strokes chin', 'person_hugs_stuffed_animal': 'Hugs stuffy', 'person_crawls_out_from_under_vehicle': 'Crawls', 'person_puts_on_hoodie': 'Puts on hoodie', 'person_takes_off_hoodie': 'Takes off hoodie', 'person_searches_in_box': 'Searches in box', 'person_adjusts_thermostat': 'Adjusts', 'person_stirs_mug': 'Stirs', 'person_opens_oven_door': 'Opens oven door', 'person_closes_oven_door': 'Closes oven door', 'person_burns_hand': 'Burns hand', 'person_searches_in_couch': 'Searches couch', 'person_salutes': 'Salutes', 'person_crates_dog': 'Crates dog', 'person_uncrates_dog': 'Uncrates dog', 'person_bounces_ball_on_floor': 'Bounces ball on floor', 'person_takes_off_pants': 'Takes off pants', 'person_puts_object_into_oven': 'Puts object into oven', 'person_takes_object_from_oven': 'Takes object from oven', 'person_searches_under_bed': 'Searches', 'person_attaches_leash_to_dog': 'Attaches leash to dog', 'person_unattaches_leash_from_dog': 'Unattaches leash', 'person_takes_down_smoke_detector': 'Takes down', 'person_puts_up_smoke_detector': 'Puts up', 'person_holds_object_above_head': 'Holds object up', 'person_rubs_foot': 'Rubs foot', 'person_gestures_listen_closely': 'Gestures listen closely', 'person_gestures_blow_kiss': 'Gestures blow kiss', 'person_dries_hair_with_hairdryer': 'Blow dries hair', 'person_gestures_cut': 'Gestures cut', 'person_gestures_watch_closely': 'Gestures watch closely', 'person_gestures_be_quiet': 'Gestures be quiet', 'person_blows_nose': 'Blows nose', 'person_sticks_out_tongue': 'Sticks out tongue', 'person_dries_hair_with_towel': 'Towel dries hair', 'person_gestures_arms_x': 'Gestures arms X', 'person_gestures_behind_me': 'Gestures behind me', 'person_pats_head': 'Pats head', 'person_puts_on_necktie': 'Puts on necktie', 'person_takes_off_necktie': 'Takes off necktie', 'person_paints_fingernails': 'Paints fingernails', 'person_wraps_box': 'Wraps box', 'person_carries_person_on_shoulders': 'Carries person on shoulders', 'person_braids_hair_of_person': 'Braids hair', 'person_carries_person_on_back': 'Carries person on back', 'person_gestures_hang_loose': 'Gestures hang loose', 'person_throws_object_into_air': 'Throws object into air', 'person_gestures_zoom_in': 'Gestures zoom in', 'person_gestures_zoom_out': 'Gestures zoom out', 'person_gestures_number_four': 'Gestures number four', 'person_brushes_hair_of_person': 'Brushes hair of person', 'person_gestures_heart': 'Gestures heart', 'person_gestures_number_one': 'Gestures number one', 'person_gestures_number_five': 'Gestures number five', 'person_gestures_number_three': 'Gestures number three', 'person_gestures_call_me': 'Gestures call me', 'person_gestures_stop': 'Gestures stop', 'person_walks_tiptoe': 'Walks tiptoe', 'person_pours_coffee_into_mug': 'Pours coffee', 'person_hugs_person_from_behind': 'Hugs from behind', 'person_opens_clothes_washer': 'Opens clothes washer', 'person_closes_clothes_washer': 'Closes clothes washer', 'person_pours_into_bowl': 'Pours into bowl', 'person_searches_in_kitchen_drawer': 'Searches in kitchen drawer', 'person_puts_down_object_on_floor': 'Puts down object on floor', 'person_falls_while_standing': 'Falls while standing', 'person_opens_suitcase': 'Opens suitcase', 'person_closes_suitcase': 'Closes suitcase', 'person_opens_kitchen_drawer': 'Opens kitchen drawer', 'person_closes_kitchen_drawer': 'Closes kitchen drawer', 'person_turns_off_faucet': 'Turns off faucet', 'person_takes_object_from_cabinet': 'Takes object from cabinet', 'person_takes_object_from_kitchen_drawer': 'Takes object from kitchen drawer', 'person_puts_object_into_kitchen_drawer': 'Puts object into kitchen drawer', 'person_turns_on_faucet': 'Turns on faucet', 'person_texts_on_phone_while_sitting': 'Texts on phone while sitting', 'person_exercises_with_plank': 'Exercises with plank', 'person_takes_object_from_bag': 'Takes object from bag', 'person_puts_object_in_cabinet': 'Puts object in cabinet', 'person_hits_person_with_pillow': 'Hits with pillow', 'person_takes_object_from_basket': 'Takes object from basket', 'person_puts_object_into_basket': 'Puts object into basket', 'person_catches_dropped_object': 'Catches dropped object', 'person_eats_apple': 'Eats apple', 'person_eats_banana': 'Eats banana', 'person_puts_object_into_bag': 'Puts object into bag', 'person_opens_dishwasher': 'Opens dishwasher', 'person_closes_dishwasher': 'Closes dishwasher', 'person_leaves_scene_through_structure': 'Leaves scene', 'person_feeds_cat': 'Feeds cat', 'person_points_to_dog': 'Points to dog', 'person_throws_object_to_dog': 'Throws object to dog', 'person_feeds_dog': 'Feeds dog', 'person_embraces_sitting_person': 'Embraces sitting person', 'person_holds_hand': 'Holds hand', 'person_shakes_hand': 'Shakes hand', 'person_takes_object_from_box': 'Takes object from box', 'person_puts_object_into_box': 'Puts object into box', 'person_cleans_eyeglasses': 'Cleans eyeglasses'}

        if modelfile is not None:
            self._load_trained(modelfile)
        else:
            self._load_pretrained()
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes())
        

    #---- <LIGHTNING>
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)                
        #return {'val_loss': avg_loss, 'avg_val_loss': avg_loss}   # as of 9.1, this does not return anything
    #---- </LIGHTNING>
        
    def totensor(self, v=None, training=False, validation=False, show=False, doflip=False, asjson=False):
        """Return captured lambda function if v=None, else return tensor"""    
        assert v is None or isinstance(v, vipy.video.Scene), "Invalid input"
        f = (lambda v, num_frames=self._num_frames, input_size=self._input_size, mean=self._mean, std=self._std, training=training, validation=validation, show=show, classname=self.__class__.__name__:
             PIP_370k._totensor(v, training, validation, input_size, num_frames, mean, std, noflip=['car_turns_left', 'car_turns_right', 'vehicle_turns_left', 'vehicle_turns_right', 'motorcycle_turns_left', 'motorcycle_turns_right'], show=show, doflip=doflip, asjson=asjson, classname=classname))
        return f(v) if v is not None else f
    

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
    def __init__(self, stride=3, activities=None, gpus=None, batchsize=None, mlbl=False, mlfl=False, modelfile=None):
        assert modelfile is not None, "Contact <info@visym.com> for access to non-public model files"

        super().__init__(pretrained=False, modelfile=modelfile, mlbl=mlbl, mlfl=mlfl)
        self._stride = stride
        self._allowable_activities = {k:v for (k,v) in [(a,a) if not isinstance(a, tuple) else a for a in activities]} if activities is not None else {k:k for k in self.classlist()}
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

    def softmax(self, x_logits):
        """Return a list of lists [(class_label, float(softmax), float(logit) ... ] for all classes and batches"""
        yh = x_logits.detach().cpu().numpy()        
        yh_softmax = F.softmax(x_logits, dim=1).detach().cpu()
        return [[(self.index_to_class(j), float(sm[j]), float(s[j])) for j in range(len(sm))] for (s,sm) in zip(yh, yh_softmax)]

    def finalize(self, vo, trackconf=None, activityconf=None, startframe=None, endframe=None):
        """In place filtering of video to finalize"""
        assert isinstance(vo, vipy.video.Scene)

        tofinalize = set([ai for (ai,a) in vo.activities().items() if (endframe is None or a.endframe() <= endframe) and (startframe is None or a.endframe() >= startframe)])
        tofinalize = tofinalize.union([ti for (ti,t) in vo.tracks().items() if ((endframe is None or t.endframe() <= endframe) and (startframe is None or t.endframe() >= startframe)) or any([ti == vo.activities(id=ai).actorid() for ai in tofinalize])])

        # Bad tracks:  Remove low confidence or too short non-moving tracks, and associated activities
        # - will throw exception that 'vo referenced before assignment' if one loop did not succceed
        if trackconf is not None:
            vo.trackfilter(lambda t: t.id() not in tofinalize or len(t)>=vo.framerate() and (t.confidence() >= trackconf or t.startbox().iou(t.endbox()) == 0)).activityfilter(lambda a: a.id() not in tofinalize or a.actorid() in vo.tracks())  
        
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
        f_logistic = lambda x,b,s=1.0: float(1.0 / (1.0 + np.exp(-s*(x + b))))
        vo.activitymap(lambda a: a.confidence(a.confidence()*f_logistic(a.attributes['logit'], -1.5)) if a.id() in tofinalize else a)
        
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
            
    def __call__(self, vi, activityiou=0.1, mirror=False, minprob=0.04, trackconf=0.2, maxdets=105, avgdets=70, throttle=True, buffered=True, finalized=True):
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
                        self.finalize(vo, trackconf=trackconf, startframe=k-finalized-5, endframe=k-5)  
                        
                    yield vo

        except Exception as e:                
            raise

        finally:
            if not (finalized is False):
                self.finalize(vo, trackconf=trackconf) if finalized == True else self.finalize(vo, trackconf=trackconf, startframe=(k//finalized)*finalized-4, endframe=k)


class ActivityTrackerCap(ActivityTracker, CAP):
    pass

    
