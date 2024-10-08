#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    # processors['clip_train'] = import_class('processor.clip_train.CLIP_Train_Processor')
    # processors['skele_clip'] = import_class('processor.skele_clip.M_Processor')
    # processors['skele_clip_prototype'] = import_class('processor.skele_clip_prototype.M_Processor')
    # processors['crossdt'] = import_class('processor.crossdt.M_Processor')
    # processors['recognition_uda'] = import_class('processor.recognition_uda.REC_Processor')
    # processors['recognition_uda_v'] = import_class('processor.recognition_uda_v.REC_Processor')
    # processors['recognition_uda_adv'] = import_class('processor.recognition_uda_adv.REC_Processor')
    # processors['recognition_uda_adv_center'] = import_class('processor.recognition_uda_adv_center.REC_Processor')
    # processors['recognition_uda_rvtclr'] = import_class('processor.recognition_uda_rvtclr.REC_Processor')
    # processors['recognition_ddtc_single'] = import_class('processor.recognition_ddtc_single.REC_Processor')
    # processors['recognition_uda_adv_pl'] = import_class('processor.recognition_uda_adv_pl.REC_Processor')
    processors['recognition_ddtc'] = import_class('processor.recognition_ddtc.REC_Processor')
    # processors['recognition_adamoco'] = import_class('processor.recognition_adamoco.REC_Processor')
    # processors['recognition_uda_coral'] = import_class('processor.recognition_uda_coral.REC_Processor')

    # processors['recognition_uda_jm'] = import_class('processor.recognition_uda_jm.REC_Processor')
    # processors['recognition_udav2'] = import_class('processor.recognition_udav2.REC_Processor')
    # processors['recognition_udav3'] = import_class('processor.recognition_udav3.REC_Processor')
    # processors['recognition_unistr'] = import_class('processor.recognition_unistr.REC_Processor')
    # processors['recognition_uda_t'] = import_class('processor.recognition_uda_t.REC_Processor')
    # processors['recognition_ss'] = import_class('processor.recognition_ss.REC_Processor')
    # processors['recognition_sd'] = import_class('processor.recognition_sd.REC_Processor')
    # processors['recognition_uda_ss'] = import_class('processor.recognition_uda_ss.REC_Processor')
    # processors['recognition_uda_mcd'] = import_class('processor.recognition_uda_mcd.REC_Processor')

    # processors['skele_clip-vae'] = import_class('processor.skele_clip-vae.M_Processor')
    # processors['skele_clip-vae-a'] = import_class('processor.skele_clip-vae-a.M_Processor')
    # processors['skele_clip_train_v3'] = import_class('processor.skele_clip_train_v3.CLIP_Train_Processor')
    # processors['skele_clip_gan'] = import_class('processor.skele_clip_gan.M_Processor')
    # processors['skele_clip_gan'] = import_class('processor.skele_clip_gan_v1.M_Processor')
    # processors['skele_clip_vaegan'] = import_class('processor.skele_clip_vaegan.M_Processor')
    # processors['skele_clip_crossvaegan'] = import_class('processor.skele_clip_crossvaegan.M_Processor')
    # processors['skele_clip_zsl'] = import_class('processor.skele_clip_zsl.M_Processor')
    # processors['skele_clip_zsl_vaegan'] = import_class('processor.skele_clip_zsl_vaegan.M_Processor')
    # processors['skele_clip_zsl_sl'] = import_class('processor.skele_clip_zsl_sl.M_Processor')
    # processors['skele_clip_zsl_a'] = import_class('processor.skele_clip_zsl_a.M_Processor')
    # processors['skele_clip_train_v4'] = import_class('processor.skele_clip_train_v4.CLIP_Train_Processor')
    # processors['demo_old'] = import_class('processor.demo_old.Demo')
    # processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    # processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
