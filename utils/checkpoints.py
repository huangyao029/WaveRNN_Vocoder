import torch

from utils.paths import Paths


def get_checkpoint_paths(checkpoint_type: str, paths : Paths):
    
    if checkpoint_type == 'voc':
        weights_path = paths.voc_latest_weights
        optim_path = paths.voc_latest_optim
        checkpoint_path = paths.voc_checkpoints
    else:
        raise NotImplementedError
    
    return weights_path, optim_path, checkpoint_path


def save_checkpoint(checkpoint_type : str, paths : Paths, model, optimizer, *,
                    name = None, is_silent = False):
    
    def helper(path_dict, is_named):
        
        s = 'named' if is_named else 'latest'
        num_exist = sum(p.exists() for p in path_dict.values())
        
        if  num_exist not in (0, 2):
            raise FileNotFoundError(
                f'We expected either both or no files in the {s} checkpoint to '
                'exist, but instead we got exactly one!'
            )
            
        if num_exist == 0:
            if not is_silent:
                print(f'Creating {s} checkpoint...')
            for p in path_dict.values():
                p.parent.mkdir(parents = True, exist_ok = True)
        else:
            if not is_silent:
                print(f'Saving to existing {s} checkpoint...')
                
        if not is_silent:
            print(f'Saving {s} weights : {path_dict["w"]}')
        
        model.save(path_dict['w'])
        
        if not is_silent:
            print(f'Saving {s} optimizer state : {path_dict["o"]}')
        
        torch.save(optimizer.state_dict(), path_dict['o'])
        
        
    weights_path, optim_path, checkpoint_path = get_checkpoint_paths(checkpoint_type, paths)
    
    latest_paths = {'w' : weights_path, 'o' : optim_path}
    helper(latest_paths, False)
    
    if name:
        named_paths = {
            'w' : checkpoint_path/f'{name}_weights.pyt',
            'o' : checkpoint_path/f'{name}_optim.pyt'
        }
        helper(named_paths, True)


def restore_checkpoint(checkpoint_type : str, paths : Paths, model, optimizer, *,
                       name = None, create_if_missing = False):
    
    weights_path, optim_path, checkpoint_path = get_checkpoint_paths(checkpoint_type, paths)
    
    if name:
        path_dict = {
            'w' : checkpoint_path/f'{name}_weights.pyt',
            'o' : checkpoint_path/f'{name}_optim.pyt',
        }
        s = 'named'
    else:
        path_dict = {
            'w' : weights_path,
            'o' : optim_path
        }
        s = 'latest'
        
    num_exist = sum(p.exists() for p in path_dict.values())
    if num_exist == 2:
        print(f'Restoring from {s} checkpoint...')
        print(f'Loading {s} weights: {path_dict["w"]}')
        model.load(path_dict['w'])
        print(f'Loading {s} optimizer state : {path_dict["o"]}')
        optimizer.load_state_dict(torch.load(path_dict['o']))
    elif create_if_missing:
        save_checkpoint(checkpoint_type, paths, model, optimizer, name = name, is_silent = False)
    else:
        raise FileNotFoundError(f'The {s} checkpoint could not be found!')
        
        