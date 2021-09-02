#!/usr/bin/env bash
SRLPATH=./data
TAGGERPATH=..



export PYTHONPATH=$TAGGERPATH:$PYTHONPATH
export PERL5LIB=$SRLPATH/lib:$PERL5LIB
export PATH=$SRLPATH/bin:$PATH


python $TAGGERPATH/scripts/eval_bert_binary.py \
  /home/yzc/hxq/edit2/scripts/models/FC_model_KE.ckpt \
 ../res \
  hyper  \



