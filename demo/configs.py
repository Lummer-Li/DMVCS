def get_config(args):
    """
    Configure training parameters for specific datasets

    Parameters
    ----------
    args : object
        Configuration object containing model parameters.
        Requires 'dataset', 'train_epochs' and 'valid_epochs' attributes

    Returns
    -------
    object
        Modified configuration object with dataset-specific settings

    Raises
    ------
    NotImplementedError
        If requested dataset is not in supported configurations
    """
    if args.dataset == 'uci-digit':
        args.train_epochs = 500
        args.valid_epochs = 100
    elif args.dataset == 'ALOI':
        args.train_epochs = 500
        args.valid_epochs = 100
    elif args.dataset == 'handwritten':
        args.train_epochs = 500
        args.valid_epochs = 100
    elif args.dataset == 'Caltech-5V':
        args.train_epochs = 500
        args.valid_epochs = 100
    elif args.dataset == 'ALOI100':
        args.train_epochs = 500
        args.valid_epochs = 100
    
    return args