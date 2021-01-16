from pmutils.kr_stopwords import STOPWORDS
from notebook import notebookapp
from collections import Counter
from datetime import datetime
import urllib
import errno
import logging
import platform
import getpass
import keyring
import json
import os
import subprocess
import ipykernel
import re
import sys
import time


# These characters are converted to unicode values that are YAML doesn't support
# Will not work with knowledge repo currently.
#class cmd_colors:
#    HEADER = '\033[95m'
#    OKBLUE = '\033[94m'
#    OKGREEN = '\033[92m'
#    WARNING = '\033[93m'
#    FAIL = '\033[91m'
#    ENDC = '\033[0m'
#    BOLD = '\033[1m'
#    UNDERLINE = '\033[4m'



def get_pm_email(invalid=False, mismatch=False):
    '''
    Grabs user's poshmark email address from their keyring.
    If their email doesn't exist in the keyring then the user will be prompted to enter their email.'
    '''
    mac_user = getpass.getuser()
    email = str(keyring.get_password('poshmark_email', mac_user)).lower()

    if mismatch:
        prompt = 'Your email inputs did not match. Please try again. Enter your poshmark email address: '
    elif invalid:
        prompt = '''Invalid Email.
        Your email should look similar to example@poshmark.com
        Please enter your poshmark email address: '''.replace('\t','')
    else:
        prompt = 'Email not found in keyring. Please enter your poshmark email address: '

    def valid_email(x):
        return x.endswith('@poshmark.com')

    if not valid_email(email):
        email = getpass.getpass(prompt=prompt).lower()
        if not valid_email(email):
            return get_pm_email(invalid=True)
        confirmation = getpass.getpass(prompt='Please confirm your poshmark email: ').lower()
        if email != confirmation:
            return get_pm_email(mismatch=True)

        keyring.set_password('poshmark_email', mac_user, email)

    return email


def knowledge_user():
    return get_pm_email().split('@')[0].capitalize()



def get_pm_email(invalid=False, mismatch=False):
    '''
    Grabs user's poshmark email address from their keyring.
    If their email doesn't exist in the keyring then the user will be prompted to enter their email.'
    '''
    mac_user = getpass.getuser()
    email = str(keyring.get_password('poshmark_email', mac_user)).lower()

    if mismatch:
        prompt = 'Your email inputs did not match. Please try again. Enter your poshmark email address: '
    elif invalid:
        prompt = '''Invalid Email.
        Your email should look similar to example@poshmark.com
        Please enter your poshmark email address: '''.replace('\t','')
    else:
        prompt = 'Email not found in keyring. Please enter your poshmark email address: '

    def valid_email(x):
        return x.endswith('@poshmark.com')

    if not valid_email(email):
        email = getpass.getpass(prompt=prompt).lower()
        if not valid_email(email):
            return get_pm_email(invalid=True)
        confirmation = getpass.getpass(prompt='Please confirm your poshmark email: ').lower()
        if email != confirmation:
            return get_pm_email(mismatch=True)

        keyring.set_password('poshmark_email', mac_user, email)

    return email



def knowledge_user():
    return get_pm_email().split('@')[0].capitalize()



def notebook_path():
    """Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    for srv in notebookapp.list_running_servers():
        try:
            if srv['token']=='' and not srv['password']:  # No token and no password, ahem...
                req = urllib.request.urlopen(srv['url']+'api/sessions')
            else:
                req = urllib.request.urlopen(srv['url']+'api/sessions?token='+srv['token'])
            sessions = json.load(req)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return os.path.join(srv['notebook_dir'],sess['notebook']['path'])
        except:
            pass  # There may be stale entries in the runtime directory
    return None


# Useful notebook details
NOTEBOOKPATH = notebook_path()
NOTEBOOKNAME = notebook_path().split('/')[-1]
NOTEBOOKDIR = "/".join(notebook_path().split('/')[:-1])


# checks for a subdirectory inside a directory
def check_for_sub_directory(path=None, subdir=None):

    if type(path) == str:
        path_list = path.split('/')
    elif type(path) == list:
        path_list = path
        path = '/'.join(path_list)
    else:
        raise ValueError('''Invalid path parameter:
        path must be a directory string or a list of directories''')

    sub_directories = os.listdir(path)
    if subdir in sub_directories:
        return path + f'/{subdir}'
    else:
        raise ValueError("{subdir} directory was not found in {dir}".format(subdir=subdir, dir=path_list[-1]))



# get the current notebook path and find analytics.
# general case: use top down search. if analytics occurs more than once, use bottom up search.
# check for knowledge_repo directory inside analytics.
def repo_path():
    directories = [x.lower() for x in NOTEBOOKPATH.split('/')]
    reversed_directories = directories.copy()
    reversed_directories.reverse()
    checklist = ['analytics','user_folder','knowledge_repo']
    directory_counts = Counter(directories)
    directory_length = len(directories)

    if directory_counts.get('analytics',0) == 1:
        # general case
        index = directories.index('analytics')
        path = NOTEBOOKPATH.split('/')[0:index+1]
        return check_for_sub_directory(path=path, subdir='knowledge_repo')

    elif directory_counts.get('analytics',0) == 0:
        # outside of analytics path - raise error
        raise ValueError("Must be working inside analytics git repository")
    else:
        index = reversed_directories.index('user_folders')+1
        if directory_length > index:
            if reversed_directories[index] == 'analytics':
                index = (-1*index)-1
                path = NOTEBOOKPATH.split('/')[:index]
                return check_for_sub_directory(path=path, subdir='knowledge_repo')
        else:
            raise ValueError("Can't locate knowledge_repo directory")

        """ TODO -- add more robust path checking
        if len(reversed_directories) >= 3:
            if reversed_directories[1] == 'user_folders'
                and reversed_directories[2] == 'analytics':
                    path = NOTEBOOKPATH.split('/')[:-3]
                    return check_for_sub_directory(path=path, subdir='knowledge_repo')
            elif
        """

    return directory_counts



# normalizes tags
def knowledge_repo_tags(filename, extra_tags=None, auto_tags=False):
    tags = ['analytics']
    if type(extra_tags) == str:
        extra_tags = [extra_tags]
    elif type(extra_tags) != list:
        extra_tags = []

    if auto_tags:
        file_tags = list(set([x for x in re.sub('[^a-zA-Z0-9]+', ' ', filename.split('.')[0].lower()).split()
                        if (len(x) > 3)
                        and x not in ['untitled','copy','scratch','test']
                        and x[:-1] not in ['untitled','copy','scratch','test']
                        and x[:-2] not in ['untitled','copy','scratch','test']
                       ]))
        #return list(set(tags + extra_tags + file_tags))
    else:
        file_tags = []
    tags = [re.sub('[^a-zA-Z0-9]+', '', t).lower() for t in set(tags + extra_tags + file_tags)]
    tags = [t for t in tags if t not in STOPWORDS]
    return list(set(tags))


# previous version
"""
def knowledge_repo_authors(authors=None, lower=True):
    author = getpass.getuser()
    if type(authors) == str:
        authors = [authors]
    elif type(authors) != list:
        authors = []
    if len(authors) < 1:
        authors = ['analytics', author]
    # return [x.strip().lower() for x in authors]
"""

# TODO allow for list of emails to be confirmed.
def knowledge_repo_authors(authors=None, lower=True):
    return [knowledge_user()]



# returns a list of headers lines to add to the notebook
def knowledge_repo_header(filename,
                          title=None,
                          extra_tags=None,
                          auto_tags=True,
                          authors=None,
                          tldr=None,
                          private=None,
                          allowed_groups=None):
    if title is None:
        title = filename.split('.')[0]

    if tldr is None:
        tldr = 'click for details'

    if type(tldr) is str:
        if len(tldr.split(' ')) < 4:
            tldr = 'click for details'

    tldr = tldr.replace('\n',' ')

    authors = knowledge_repo_authors(authors=authors)
    tags = knowledge_repo_tags(filename, extra_tags=extra_tags, auto_tags=auto_tags)

    # todo -- add checks for previous KR posts for accurate created_at vs updated_at.
    if sys.platform == 'darwin' and False:
        created_at = str(datetime.fromtimestamp(os.stat(filename).st_birthtime))[:19]
    else:
        created_at = str(datetime.fromtimestamp(os.path.getmtime(filename)))[:19]
    updated_at = str(datetime.now())[:19]

    if tldr is None:
        tldr = ''

    if private:
        private = 'private: true'
    else:
        private = ''

    if allowed_groups is None:
        # todo
        allowed_groups = None

    # todo improve header_content readability
    """
    header_content = ['--- \n',
                      f'title: {title} \n'
                      'authors: \n'] \
                      + [' -' + '\n -'.join(authors)] \
                      + ['tags: \n'],
                      + [f'created_at: {created_at} \n',
                          f'updated_at: {updated_at} \n',
                          f'tldr: {tldr} \n',
                          private,
                         '---']"""

    header_content = \
f"""---
title: {title}
authors:
"""+ '- ' + '\n- '.join(authors) + "\ntags:\n" + '- ' + '\n- '.join(tags) + f"""
created_at: {created_at}
updated_at: {updated_at}
tldr: {tldr}
{private}
"""
    return [x +' \n' for x in header_content.split('\n')] + ['---']



# makes a directory and waits up to 10 seconds for it to exists
def create_project(path=None, wait=10):
    os.makedirs(path, exist_ok=True)
    wait_for_path(path=path, wait=wait)
    add_to_git(path)



# waits for knowledge notebook file to exist before commiting
def wait_for_path(path=None, wait=10, strict=False):
    loops = int(wait/0.1)
    for i in range(loops):
        time.sleep(0.1)
        if os.path.exists(path):
            return
    # TODO: add option for raising file not found error
    if strict:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def add_assets(filepath, asset_path):

    with open(asset_path,'rb') as rf:
        with open(filepath + '/' + asset_path,'wb+') as wf:
            wf.write(rf.read())



def add_to_git(file_path):
    try:
        subprocess.run( ['git','add',filepath])
    except:
        return



def auto_commit_knowledge_notebook(notebook_path):
    add_to_git(notebook_path)
    commit_cmds = ['git', 'commit', '-m', 'adding notebook to knowledge repo', notebook_path]
    push_cmds = ['git', 'push']
    wait_for_path(path=notebook_path)
    commit_results = subprocess.run(commit_cmds, capture_output=True)
    if commit_results.returncode == 0:
         push_results = subprocess.run(push_cmds, capture_output=True)
    else:
        push_results = 'commits were not pushed'

    try:
        if commit_results.returncode == 0:
            if push_results.returncode != 0:
                raise Exception("""
        --- Error commiting notebook ---
        """.upper()\
        + """
        Return Code: """ + """{}
        """.format('\n\t'+str(push_results.returncode))
        + """
        Command: """ + """{}
        """.format('\n\t'+str(' '.join(push_results.args)))
        + """
        Error Message: """ + """{}
        """.format('\n\t'+'\n\t'.join(str(push_results.stderr.decode('ascii')).split('\n')))
        )
        else:
            raise Exception("""
        --- Error commiting notebook ---
        """.upper()\
        + """
        Return Code: """ + """{}
        """.format('\n\t'+str(commit_results.returncode))
        + """
        Command: """ + """{}
        """.format('\n\t'+str(' '.join(commit_results.args)))
        + """
        Error Message: """ + """{}
        """.format('\n\t'+'\n\t'.join(str(commit_results.stderr.decode('ascii')).split('\n')))
        )

        auto_commit_success_msg = \
        ("""
        --- Successfully commited notebook ---
        """.upper()
        + """{}
        """.format('\n\t'+'\n\t'.join(str(commit_results.stdout.decode('ascii')).split('\n')))
        + """
        --- Successfully pushed notebook ---
        """.upper()
        + """{}
        """.format('\n\t'+'\n\t'.join(str(push_results.stdout.decode('ascii')).split('\n')))
        + """* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        The KR version of this notebook was created and successfully pushed to github.
        Your post should appear in Knowledge Repo momentairly.
        """
        )
        print(auto_commit_success_msg)

    except:
        logging.exception("""
    The KR version of this notebook was created but autocommit was unsuccessful.
    You will need to manually commit and push changes to finish creating your Knowledge Repo post.
    For details on the error. See the logging message below:
    """)

SUMMARY_MSG = """
Please write a short summary of your knowledge post. This will appear on the post feed.
To avoid this prompt -- pass a summary parameter into the function: add_knowledge(summary=my_summary)
"""


# adds a formated header cell to a copy of the notebook
# notebook copy is written to knowledge_repo.
def add_knowledge(filename=NOTEBOOKNAME,
                   title=None,
                   authors=[knowledge_user()],
                   extra_tags=None,
                   auto_tags=True,
                   summary=None,
                   project='stash',
                   header=False,
                   update=True,
                   private=False,
                   allowed_groups=None,
                   image_assets=list(),
                   auto_commit=False):

    if filename is None:
        raise ValueError("filename=None is invalid. A filename must be provided.")
        return

    if summary is None:
        summary = input(SUMMARY_MSG)

    if not header:
        header = knowledge_repo_header(filename.split('/')[-1],
                                        title=title,
                                        extra_tags=extra_tags,
                                        auto_tags=auto_tags,
                                        authors=authors,
                                        tldr=summary,
                                        private=private,
                                        allowed_groups=allowed_groups)
        header_format = dict()
        header_format['cell_type'] = 'raw'
        header_format['metadata'] = dict()
        header_format['source'] = header

        with open(filename, 'r') as f:
            temp = json.loads(f.read())
        temp['cells'].insert(0,header_format)

    else:
        with open(filename, 'r') as f:
            # remove unicode to prevent yaml errors
            temp = json.loads(re.sub(r'[^\x00-\x7F]+','', f.read()))

    output_path = repo_path() +'/'+ project + '/' + filename.split('.')[0]
    create_project(path=output_path)
    output_file = output_path + '/' + filename.replace('.ipynb','_kr.ipynb').split('/')[-1]

    with open(output_file, 'w+') as f:
        json.dump(temp,f)
        if auto_commit: # TODO
            print('auto commit is still in development. Please manually commit this post.')
            #auto_commit_knowledge_notebook(output_file)
        #else: TODO
            #print('{} has been created.'.format(output_file.split('/')[-1])
            #    + '\nPlease commit and push this file manually or rerun add_knowledge with '
            #    + 'auto_commit=True')


    for asset in image_assets:
        asset_name = asset.split('/')[-1]
        kr_asset = output_path + '/' + asset_name
        add_assets(output_path, asset)

        if auto_commit:
            auto_commit_knowledge_notebook(kr_asset)
        else:
            print('{} has been created.'.format(kr_asset)
                + '\nPlease commit and push this file manually or rerun add_knowledge with '
                + 'auto_commit=True')
