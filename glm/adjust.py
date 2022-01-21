
import torch
#path_old='/workspace/xuzou/codex-glm/192000/mp_rank_00_model_states.pt'
#path_new='/workspace/xuzou/glm-10b-code/192000/mp_rank_00_model_states.pt'
path_old='/workspace/xuzou/glm_pts2/blocklm-10b-chinese07-08-15-28/83000/mp_rank_00_model_states.pt'
path_new='/workspace/xuzou/glm_pts2/83000/mp_rank_00_model_states.pt'
old=torch.load(path_old, map_location='cpu')

old['module']['transformer.word_embeddings.weight'] = old['module']['word_embeddings.weight']
del old['module']['word_embeddings.weight']

old['module']['mixins.block_position_embedding.block_position_embeddings.weight'] = old['module']['transformer.block_position_embeddings.weight']
del old['module']['transformer.block_position_embeddings.weight']
# replace names, mixins index to keys
# oldm = old['module']
# for k in list(oldm.keys()):
#     if k.startswith('mixins.0'):
#         new_k = k.replace('mixins.0', 'mixins.extra_position_embedding')
#     elif k.startswith('mixins.1'):
#         new_k = k.replace('mixins.1', 'mixins.attention_plus')
#     else:
#         continue
#     oldm[new_k] = oldm[k]
#     del oldm[k]
# save to destination



# %%
torch.save(old, path_new)

