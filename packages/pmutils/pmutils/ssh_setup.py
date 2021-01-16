""" -- WIP --
    This package will eventually handle all of the core ssh needs of our team.
    Current functionality is limited and only includes creating a config file
    calling ssh-add.
    Future functionality will include creating ssh id_rsa tokens, sending public keys,
    and automatically calling ssh-add on pc login.
"""

import subprocess
import getpass


USER = getpass.getuser()


# Create config file in .ssh/ directory and call ssh-add
# TODO create a backup of an existing config file if present.
def ssh_keychain_setup(path=f'/Users/{USER}/.ssh/'):
    user = getpass.getuser()
    config_path = path+'config'
    config_content = ["Host *", "\n", "UseKeychain yes", "\n"]

    with open(config_path, 'w+') as config:
        config.writelines(config_content)
        print('created {path}'.format(path=config_path))
    subprocess.call(['ssh-add', '-K', '-t', '52w'])
    
