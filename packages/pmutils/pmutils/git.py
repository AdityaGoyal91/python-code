import subprocess
import getpass
import os


USER = getpass.getuser()
CWD = os.getcwd()
ANALYTICS = CWD.split('/Analytics')[0] + '/Analytics'



# runs git pull orgin for the specified repository and returns process output.
# output can be ignored by specifying the parameter _return=False
# user needs to have ssh access to the specified remote repository
# subprocess prompts for password if there is not active ssh-agent
def pull(repo=ANALYTICS, subdir=None, _return=True):

    if subdir:
        repo = '/'.join([repo] + subdir)

    cmd = ['git', '-C', repo, 'pull', 'origin']
    p = subprocess.run(cmd, capture_output=True)

    #TODO: add error handling for p.returncode != 0 (failed pull request)
    if _return:
        return p



# checks analytics repository every 10 seconds
if __name__ == '__main__':
    while True:
        pull(repo=ANALYTICS, _return=False)
        time.sleep(10)
