def log(verbose = False):

    def decorators(F):

        def wrapper(*args, **kwargs):

            if verbose:

                print('[*] Enter => {}'.format(F.__name__))

            result = F(*args, **kwargs)
            
            if verbose:

                print('[*] Exit => {}'.format(F.__name__))

            return result

        return wrapper

    return decorators


