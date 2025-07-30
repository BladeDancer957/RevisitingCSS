from segmentation_module import make_model
import argparser
import torch
import numpy as np
import os

if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    if len(opts.step) == 1:
        opts.step = opts.step[0]
    else:
        exit()

    opts.inital_nb_classes = [opts.num_classes]
    model = make_model(opts, classes=opts.inital_nb_classes)

    device = torch.device(opts.local_rank)
    # model = DistributedDataParallel(model.cuda(device))
        
    task_name = f"{opts.task}-{opts.dataset}"
    # import pdb; pdb.set_trace()
    ckpt = f"{opts.checkpoint}/{task_name}_{opts.name}_{opts.step}.pth"
    checkpoint = torch.load(ckpt, map_location="cpu")
    # model.load_state_dict(checkpoint["model_state"])
    state_dict = checkpoint["model_state"]
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print(model.cls)
    save_flag = f"{opts.dataset}_{opts.name}_{opts.task}_{opts.step}"
    print(f"save flag is {save_flag}")
    class_embedding = model.cls[0].weight.squeeze(-1).squeeze(-1).detach().cpu().numpy()
    print(class_embedding.shape)
    os.makedirs(f"{opts.backbone}_class_embedding_probing", exist_ok=True)
    np.save(f"{opts.backbone}_class_embedding_probing/{save_flag}.npy", class_embedding)