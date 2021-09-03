import argparse
import logging
import os
import pickle
from copy import deepcopy
import json
import torch
from tqdm.auto import tqdm

from src.data.binary_augmented_kilt import BinaryAugmentedKILT
from src.models.bert_binary_augmented_kilt import BertBinaryAugmented
from src.models.bert_binary_kilt import BertBinary
from src.utils import batch_it, shuffle_it

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="Filename of the model",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where to save files",
    )
    parser.add_argument(
        "method",
        type=str,
        choices=["baseline", "hyper","multi"],
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=["all"] + [str(i) for i in range(12)],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--from",
        type=int,
        default=0,
        dest="from_idx",
    )
    parser.add_argument(
        "--to",
        type=int,
        default=100000,
        dest="to_idx",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.method == "baseline":
        model = BertBinary.load_from_checkpoint(args.model)
        model = model.eval().to(args.device)

        model.hparams.dev_data_path='/home/yzc/hxq/edit2/datasets/fever-dev-kilt.jsonl'
        val_dataset0 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
            return_view=True,
            all_views=True,
        )
        val_dataset0 = list(shuffle_it(list(enumerate(val_dataset0))))[
            args.from_idx : args.to_idx
        ]

        val_dataset1 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
        )
        preds = torch.tensor([e["pred"] for e in val_dataset1])

        all_logits = {}
        all_rephrases = {}
        iter_ = tqdm(val_dataset0)
        for j, d0 in iter_:
            tmodel = deepcopy(model)
            optimizer = torch.optim.RMSprop(
                [
                    p
                    for n, p in tmodel.named_parameters()
                    if (
                        (args.layer != "all" and f".{args.layer}." in n)
                        or args.layer == "all"
                    )
                    and all(
                        e not in n.lower()
                        for e in (
                            "bias",
                            "norm",
                            "embeddings",
                            "classifier",
                            "pooler",
                            "shared",
                            "embed",
                            "positions",
                        )
                    )
                ],
                lr=1e-5,
            )

            while True:
                logit = tmodel(
                    {
                        k: v.to(tmodel.device)
                        for k, v in val_dataset1.collate_fn([d0]).items()
                        if isinstance(v, torch.Tensor)
                    }
                )

                if (logit > 0).item() == d0["alt"]:
                    break

                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit,
                    torch.tensor([d0["alt"]], device=tmodel.device).float(),
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            all_rephrases[j] = tmodel.sample(d0["view"]).cpu()

            all_logits_batch = []
            for i, d1 in enumerate(batch_it(val_dataset1, args.batch_size)):
                all_logits_batch.append(tmodel.sample([e["src"] for e in d1]).cpu())

            all_logits[j] = torch.cat(all_logits_batch)

            iter_.set_postfix(
                succ=sum(
                    val_dataset1[k]["alt"] == (v[k] > 0).item()
                    for k, v in all_logits.items()
                )
                / len(all_logits),
                retain=sum(
                    (
                        ((v[:k] > 0) == preds[:k]).sum()
                        + ((v[k + 1 :] > 0) == preds[k + 1 :]).sum()
                    )
                    / (len(v) - 1)
                    for k, v in all_logits.items()
                ).item()
                / len(all_logits),
                equiv=sum(
                    (v.sign() == all_logits[k][k].sign()).float().mean().item()
                    for k, v in all_rephrases.items()
                )
                / len(all_rephrases),
            )

        filename = os.path.join(
            args.output_path,
            f"all_logits-{args.from_idx}-{args.to_idx}-baseline-{args.layer}.pkl",
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_logits, f)

        filename = os.path.join(
            args.output_path,
            f"all_rephrases-{args.from_idx}-{args.to_idx}-baseline-{args.layer}.pkl",
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_rephrases, f)

    elif args.method == "hyper":
        model = BertBinaryAugmented.load_from_checkpoint(args.model)
        model.model = BertBinary.load_from_checkpoint(
            model.hparams.model_checkpoint
        ).model
        model = model.eval().to(args.device)

        val_dataset0 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path='/home/yzc/hxq/edit2/datasets/fever-dev-kilt.jsonl',#model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
            return_view=True,
            all_views=True,
        )
        val_dataset0 = list(shuffle_it(list(enumerate(val_dataset0))))[
            args.from_idx : args.to_idx
        ]

        val_dataset1 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path='/home/yzc/hxq/edit2/datasets/fever-dev-kilt.jsonl',#model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
        )
        preds = torch.tensor([e["pred"] for e in val_dataset1])

        model.val_dataloader(shuffle=False)

        all_logits = {}
        all_rephrases = {}
        right = {'s0': [], 's1': [], 'r0': [], 'r1': [], 'e0': [], 'e1': []}
        iter_ = tqdm(val_dataset0, ncols=200)
        for j, d0 in iter_:

            with torch.no_grad():
                logits_orig, params_dict = model.get_logits_orig_params_dict(
                    {
                        k: v.to(model.device)
                        for k, v in model.val_dataset.get_batch([], d0["cond"]).items()
                    }
                )
            _, logits, params_dict = model.sample(
                d0["view"],
                d0["cond"],
                logits_orig,
                params_dict,
                # stop_condition=lambda condition, logits, n_iter: (
                #                                                          ("REFUTES >> SUPPORTS" in condition and logits[
                #                                                              -1] < 0)
                #                                                          or ("SUPPORTS >> REFUTES" in condition and
                #                                                              logits[-1] > 0)
                #                                                  )
                #                                                  and n_iter < 5,
            )

            all_rephrases[j] = logits.cpu()
            all_logits_batch = []
            for i, d1 in enumerate(batch_it(val_dataset1, args.batch_size)):
                _, logits, _ = model.sample(
                    [e["src"] for e in d1], d0["cond"], logits_orig, params_dict
                )
                all_logits_batch.append(logits.cpu())

            all_logits[j] = torch.cat(all_logits_batch)

            iter_.set_postfix(
                succ=sum(
                    val_dataset1[k]["alt"] == (v[k] > 0).item()
                    for k, v in all_logits.items()
                )
                     / len(all_logits),
                retain=sum(
                    (
                            ((v[:k] > 0) == preds[:k]).sum()
                            + ((v[k + 1:] > 0) == preds[k + 1:]).sum()
                    )
                    / (len(v) - 1)
                    for k, v in all_logits.items()
                ).item()
                       / len(all_logits),
                equiv=sum(
                    (v.sign() == all_logits[k][k].sign()).float().mean().item()
                    for k, v in all_rephrases.items()
                )
                      / len(all_rephrases),
            )
        #
        for (k, v),v1 in zip(all_logits.items(),val_dataset1):


                res_label = []
                x1 = []
                ori_x1 = []
                wrong_preds = []
                for i, (vv, pp) in enumerate(zip(v[:k + 1], preds[:k + 1])):

                    if not (vv > 0) == pp and (vv > 1 or vv < -1):
                        x1.append(i)
                        wrong_preds.append(vv.tolist())
                    if not (vv > 0) == pp:
                        ori_x1.append(i)

                        res_label.append(0)
                    else:
                        res_label.append(1)
                x2 = []
                ori_x2 = []
                for i, (vv, pp) in enumerate(zip(v[k + 1:], preds[k + 1:])):

                    if not (vv > 0) == pp and (vv > 1 or vv < -1):
                        x2.append(i)
                        wrong_preds.append(vv.tolist())
                    if not (vv > 0) == pp:
                        ori_x2.append(i)

                        res_label.append(0)
                    else:
                        res_label.append(1)

                right['r0'].append(json.dumps({k:x1+x2}))#uncontain [-1,1]
                # right['r1'].append(json.dumps({k:ori_x1+ori_x2}))#contain [-1,1]
                right['e0'].append(json.dumps({k: wrong_preds}))  # after
                # right['s0'].append(json.dumps({val_dataset1[k]['src']: [val_dataset1[i]['src'] for i in x1+x2]}))  # uncontain
                # right['s1'].append(json.dumps({val_dataset1[k]['src']: [val_dataset1[i]['src'] for i in ori_x1+ori_x2]}))  # contain
        #
        filename = os.path.join(
            args.output_path, f"all_logits-{args.from_idx}-{args.to_idx}.pkl"
        )
        with open('../res/retain002.txt', "w", encoding='utf-8') as f:
           f.write(json.dumps(right, ensure_ascii=False))
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_logits, f)

        filename = os.path.join(
            args.output_path, f"all_rephrases-{args.from_idx}-{args.to_idx}.pkl"
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_rephrases, f)

    elif args.method == "multi":
        model = BertBinaryAugmented.load_from_checkpoint(args.model)
        model.model = BertBinary.load_from_checkpoint(
            model.hparams.model_checkpoint
        ).model
        model = model.eval().to(args.device)

        val_dataset0 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path='/home/yzc/hxq/edit2/datasets/fever-dev-kilt.jsonl',#model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
            return_view=True,
            all_views=True,
        )
        val_dataset0 = list(shuffle_it(list(enumerate(val_dataset0))))[
            args.from_idx : args.to_idx
        ]

        val_dataset1 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path='/home/yzc/hxq/edit2/datasets/fever-dev-kilt.jsonl',#model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
        )
        preds = torch.tensor([e["pred"] for e in val_dataset1])

        model.val_dataloader(shuffle=False)





        for mode in ['max','mean','sum','min']:#'row','max','mean','sum'
            all_logits = {}
            all_rephrases = {}
            right = {'s0': [], 's1': [], 'r0': [], 'r1': [], 'e0': [], 'e1': []}
            iter_ = tqdm(val_dataset0, ncols=200, desc=mode)
            print(mode)
            precesson, param_stack,data_stack = 0, [],[]
            for j, d0 in iter_:

                with torch.no_grad():
                    logits_orig, params_dict = model.get_logits_orig_params_dict(
                        {
                            k: v.to(model.device)
                            for k, v in model.val_dataset.get_batch([], d0["cond"]).items()
                        }
                    )

                if mode=='row':

                    for n, p in model.model.named_parameters():
                        p.data += params_dict.get(n, 0)
                if precesson != 4:
                    precesson+=1
                    param_stack.append(params_dict)
                    data_stack.append([j,d0])
                    continue
                # print()
                if mode=='max':
                #max
                    params_dict0={n:torch.max(param_stack[0][n],param_stack[1][n])  for n in param_stack[0]}
                    params_dict1={n:torch.max(param_stack[2][n],param_stack[3][n]) for n in param_stack[0]}
                    params_dict={n:torch.max(params_dict0[n],params_dict1[n]) for n in param_stack[0]}
                    params_dict1,params_dict0={},{}
                if mode=='min':
                #max
                    params_dict0={n:torch.min(param_stack[0][n],param_stack[1][n])  for n in param_stack[0]}
                    params_dict1={n:torch.min(param_stack[2][n],param_stack[3][n]) for n in param_stack[0]}
                    params_dict={n:torch.min(params_dict0[n],params_dict1[n]) for n in param_stack[0]}
                    params_dict1,params_dict0={},{}
                    # params_dict={p: torch.max(torch.stack([param_stack[i][p] for i in range(precesson)], dim=0), dim=0) for p in param_stack[0]}
                    # params_dict = {p: max(param_stack[i][p] for i in range(precesson)) for p in param_stack[0]}

                # #mean
                # params_dict = {n: torch.mean(
                #     torch.stack((param_stack[0][n], param_stack[1][p], param_stack[2][n1], param_stack[3][p1]), dim=0),
                #     dim=0) for n, p, n1, p1 in zip(param_stack[0], param_stack[1], param_stack[2], param_stack[3])}
                if mode == 'mean':
                    # params_dict = {p: torch.mean(torch.stack([param_stack[i][p] for i in range(precesson)], dim=0), dim=0) for p in param_stack[0]}
                    params_dict = {p: sum(param_stack[i][p] for i in range(precesson)) / precesson for p in param_stack[0]}


                # #sum
                # params_dict = {n: torch.sum(
                #     torch.stack((param_stack[0][n], param_stack[1][p], param_stack[2][n1], param_stack[3][p1]), dim=0),
                #     dim=0) for n, p, n1, p1 in zip(param_stack[0], param_stack[1], param_stack[2], param_stack[3])}
                # params_dict={p: torch.sum(torch.stack([param_stack[i][p] for i in range(precesson)],dim=0),dim=0) for p in param_stack[0]}
                if mode =='sum':
                    params_dict={p: sum(param_stack[i][p] for i in range(precesson)) for p in param_stack[0]}

                param_stack=[]
                precesson = 0
                for j,d0 in data_stack:
                    _, logits, params_dict = model.sample(
                        d0["view"],
                        d0["cond"],
                        logits_orig,
                        params_dict,
                        stop_condition=lambda condition, logits, n_iter: (
                            ("REFUTES >> SUPPORTS" in condition and logits[-1] < 0)
                            or ("SUPPORTS >> REFUTES" in condition and logits[-1] > 0)
                        )
                        and n_iter < 5,
                    )

                    all_rephrases[j] = logits.cpu()

                    all_logits_batch = []
                    for i, d1 in enumerate(batch_it(val_dataset1, args.batch_size)):
                        _, logits, _ = model.sample(
                            [e["src"] for e in d1], d0["cond"],logits_orig, params_dict
                        )
                        all_logits_batch.append(logits.cpu())

                    all_logits[j] = torch.cat(all_logits_batch)

                    iter_.set_postfix(
                        succ=sum(
                            val_dataset1[k]["alt"] == (v[k] > 0).item()
                            for k, v in all_logits.items()
                        )
                        / len(all_logits),
                        retain=sum(
                            (
                                ((v[:k] > 0) == preds[:k]).sum()
                                + ((v[k + 1 :] > 0) == preds[k + 1 :]).sum()
                            )
                            / (len(v) - 1)
                            for k, v in all_logits.items()
                        ).item()
                        / len(all_logits),
                        equiv=sum(
                            (v.sign() == all_logits[k][k].sign()).float().mean().item()
                            for k, v in all_rephrases.items()
                        )
                        / len(all_rephrases),
                        len_logits=len(all_logits),
                        len_rephrases=len(all_rephrases)
                    )
                data_stack=[]

        #
        # for (k, v),v1 in zip(all_logits.items(),val_dataset1):
        #
        #
        #         res_label = []
        #         x1 = []
        #         ori_x1 = []
        #         wrong_preds = []
        #         for i, (vv, pp) in enumerate(zip(v[:k + 1], preds[:k + 1])):
        #
        #             if not (vv > 0) == pp and (vv > 1 or vv < -1):
        #                 x1.append(i)
        #             if not (vv > 0) == pp:
        #                 ori_x1.append(i)
        #                 wrong_preds.append(vv.tolist())
        #                 res_label.append(0)
        #             else:
        #                 res_label.append(1)
        #         x2 = []
        #         ori_x2 = []
        #         for i, (vv, pp) in enumerate(zip(v[k + 1:], preds[k + 1:])):
        #
        #             if not (vv > 0) == pp and (vv > 1 or vv < -1):
        #                 x2.append(i)
        #             if not (vv > 0) == pp:
        #                 ori_x2.append(i)
        #                 wrong_preds.append(vv.tolist())
        #                 res_label.append(0)
        #             else:
        #                 res_label.append(1)
        #
        #         right['r0'].append(json.dumps({k:x1+x2}))#uncontain [-1,1]
        #         right['r1'].append(json.dumps({k:ori_x1+ori_x2}))#contain [-1,1]
        #         right['e0'].append(json.dumps({k: wrong_preds}))  # after
        #         right['s0'].append(json.dumps({val_dataset1[k]['src']: [val_dataset1[i]['src'] for i in x1+x2]}))  # uncontain
        #         right['s1'].append(json.dumps({val_dataset1[k]['src']: [val_dataset1[i]['src'] for i in ori_x1+ori_x2]}))  # contain
        #
        filename = os.path.join(
            args.output_path, f"all_logits-{args.from_idx}-{args.to_idx}.pkl"
        )
        # with open('../res/retain002.txt', "w", encoding='utf-8') as f:
        #    f.write(json.dumps(right, ensure_ascii=False))
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_logits, f)

        filename = os.path.join(
            args.output_path, f"all_rephrases-{args.from_idx}-{args.to_idx}.pkl"
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_rephrases, f)


#result
'''
single: 1000/1000 [11:39<00:00,  1.43it/s, equiv=0.854, retain=0.983, succ=1]
row: 1000/1000 [03:09<00:00,  5.28it/s, equiv=1, retain=0.371, succ=0.545]
sum:  1000/1000 [02:55<00:00,  5.70it/s, equiv=1, retain=0.323, succ=0.65]
mean:  1000/1000 [03:03<00:00,  5.45it/s, equiv=0.831, retain=0.819, succ=0.95]
max:  1000/1000 [02:55<00:00,  5.70it/s, equiv=0.9, retain=0.673, succ=0.98]

“pred” = orinal PREDiction from the model; “alt” = ALTernative prediction we want the model to get; “cond” = CONDition we use for the update (aka input of the hypernetwork)
for binary predictions when ‘pred’ is TRUE ,‘alt’ is FALSE and the other way around. for QA “pred” is the answer the model outputs and “alt” is the alternative prediction we want the model to make
'''

'''
row: one by one equiv=0.998, retain=0.372, succ=0.625]


  warnings.warn(_create_warning_msg(
max:   0%|                                                                                                                                                                     | 0/1000 [00:00<?, ?it/s]max
max: 100%|████████████████████████████████████████████████████████████████████████████████| 1000/1000 [09:48<00:00,  1.70it/s, equiv=0.874, len_logits=800, len_rephrases=800, retain=0.621, succ=0.899]
mean:   0%|                                                                                                                                                                    | 0/1000 [00:00<?, ?it/s]mean
mean: 100%|███████████████████████████████████████████████████████████████████████████████| 1000/1000 [10:06<00:00,  1.65it/s, equiv=0.845, len_logits=800, len_rephrases=800, retain=0.638, succ=0.892]
sum:   0%|                                                                                                                                                                     | 0/1000 [00:00<?, ?it/s]sum
sum: 100%|████████████████████████████████████████████████████████████████████████████████| 1000/1000 [10:06<00:00,  1.65it/s, equiv=0.862, len_logits=800, len_rephrases=800, retain=0.522, succ=0.812]
'''

'''
max:   0%|                                                                                                                                                                     | 0/1000 [00:00<?, ?it/s]max
max: 100%|████████████████████████████████████████████████████████████████████████████████| 1000/1000 [09:33<00:00,  1.74it/s, equiv=0.874, len_logits=800, len_rephrases=800, retain=0.621, succ=0.899]
mean:   0%|                                                                                                                                                                    | 0/1000 [00:00<?, ?it/s]mean
mean: 100%|███████████████████████████████████████████████████████████████████████████████| 1000/1000 [10:00<00:00,  1.67it/s, equiv=0.845, len_logits=800, len_rephrases=800, retain=0.638, succ=0.892]
sum:   0%|                                                                                                                                                                     | 0/1000 [00:00<?, ?it/s]sum
sum: 100%|████████████████████████████████████████████████████████████████████████████████| 1000/1000 [10:01<00:00,  1.66it/s, equiv=0.862, len_logits=800, len_rephrases=800, retain=0.522, succ=0.812]
min:   0%|                                                                                                                                                                     | 0/1000 [00:00<?, ?it/s]min
min: 100%|████████████████████████████████████████████████████████████████████████████████| 1000/1000 [09:54<00:00,  1.68it/s, equiv=0.853, len_logits=800, len_rephrases=800, retain=0.634, succ=0.891]
'''