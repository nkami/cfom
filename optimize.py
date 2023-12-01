import torch
import json
import numpy as np
from pathlib import Path
from utils import load_tokenizer_from_file, read_csv
from models import InteractionEncoder, InteractionTranslator, TransformerEncoder, TransformerDecoder
from datasets import FSMolDataSet
from train import generate_molecules
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to a trained model directory')
    args = parser.parse_args()
    model_path = args.model_path
    with open(f'{model_path}/hyper_params.json', 'r') as f:
        hyper_params = json.load(f)

    optimization_parameters = {
        'guidance_scale': hyper_params['guidance_scale'],
        'sampling_method': 'top_p',  # can be 'top_p' or 'top_k'
        'num_molecules_generated': 20,
        'p': 1.,
        'k': 40,
    }
    print(f'{optimization_parameters}')

    hyper_params.update(optimization_parameters)
    output_dir = Path(f'./results/{model_path}')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(f'{str(output_dir)}/hyper_params.json', 'w') as f:
        json.dump(hyper_params, f, indent=4)

    train_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_train.npz')
    valid_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_valid.npz')
    test_prot_embeds = np.load('./data/fsmol/prot_embeds_fsmol_test.npz')

    train_non_chiral_smiles, train_backbones, train_chains, train_assay_ids, train_types, train_labels = read_csv('./data/fsmol/train.csv')
    valid_non_chiral_smiles, valid_backbones, valid_chains, valid_assay_ids, valid_types, valid_labels = read_csv('./data/fsmol/valid.csv')
    test_non_chiral_smiles, test_backbones, test_chains, test_assay_ids, test_types, test_labels = read_csv('./data/fsmol/test.csv')
    tokenizer = load_tokenizer_from_file(f'./models/tokenizer_object.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InteractionTranslator(prot_encoder=InteractionEncoder(2 + 1024, hyper_params['embedding_dim']),
                                  mol_encoder=TransformerEncoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                                                 hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['encoder_n_head'],
                                                                 n_layers=hyper_params['encoder_n_layer'],
                                                                 max_length=hyper_params['max_mol_len'],
                                                                 pad_token=tokenizer.token_to_id('<pad>')),
                                  decoder=TransformerDecoder(len(tokenizer.get_vocab()), embedding_dim=hyper_params['embedding_dim'],
                                                             hidden_size=hyper_params['embedding_dim'], nhead=hyper_params['decoder_n_head'],
                                                             n_layers=hyper_params['decoder_n_layer'],
                                                             max_length=hyper_params['max_mol_len']))
    model.load_state_dict(torch.load(f'{model_path}/model.pt'))
    model.to(device)
    test_ds = FSMolDataSet(test_non_chiral_smiles, test_backbones, test_assay_ids, test_types, test_labels,
                           test_prot_embeds, tokenizer, calc_rf=True, use_backbone=True)
    model.eval()
    with torch.no_grad():
        for task in test_ds.tasks:
            if task.get('clf', None) is None:
                continue
            cur_interaction = torch.from_numpy(np.append(task['protein'], [1, task['assay_type']]))
            cur_interaction = cur_interaction.float().to(device)
            prot_embed = torch.reshape(model.prot_encoder(cur_interaction), (1, 1, -1))
            zero_prot_embed = torch.zeros_like(prot_embed).to(device)
            all_mol_embeds = []
            idx = 0
            while idx < len(task['inactive']):
                end = min(idx + hyper_params['bs'], len(task['inactive']))
                batch = task['inactive'][idx: end]
                batch_size = end - idx
                idx = end
                cur_mols = torch.stack(batch).to(device)
                cur_mols_embeds = model.mol_encoder(cur_mols)
                all_mol_embeds.extend([cur_mols_embeds[i, :, :].cpu() for i in range(batch_size)])

            current_sample_pairs = {}
            for orig_mol_embed, orig_mol, orig_backbone in zip(all_mol_embeds, task['inactive_smiles'],
                                                               task['inactive_backbone']):
                orig_mol_embed = orig_mol_embed.to(device)
                orig_mol_embed = torch.unsqueeze(orig_mol_embed, dim=0)
                memory = torch.concat([prot_embed, orig_mol_embed], dim=1)
                uncond_memory = torch.concat([zero_prot_embed, orig_mol_embed], dim=1)
                generated_mols = generate_molecules(model.decoder, memory, uncond_memory, device, tokenizer,
                                                    hyper_params['max_mol_len'],
                                                    tokenizer.token_to_id('<bos>'),
                                                    tokenizer.token_to_id('<eos>'),
                                                    hyper_params['guidance_scale'],
                                                    hyper_params['num_molecules_generated'],
                                                    hyper_params['sampling_method'],
                                                    hyper_params['p'],
                                                    hyper_params['k'],
                                                    orig_backbone,
                                                    orig_mol)
                current_sample_pairs[orig_mol] = generated_mols
            cur_output_dir = Path(f'{str(output_dir)}/{task["assay_id"]}')
            cur_output_dir.mkdir(parents=True, exist_ok=True)
            with open(f'{str(cur_output_dir)}/samples.txt', 'w') as f:
                lines = []
                for orig_mol in current_sample_pairs.keys():
                    for sample in current_sample_pairs[orig_mol]:
                        lines.append(f'{orig_mol} {sample}\n')
                f.writelines(lines)


