import os
import subprocess



def post_knowledge(filename=None,
                  repo='repo',
                  project='examples',
                  post=None,
                  submit=True,
                  update=True,
                  debug=False):
    if filename is None:
        raise ValueError("filename=None is invalid. A valid filename must be provided.")
        return

    if post is None:
        post = filename.replace('_kr.','_').split('/')[-1]
    post = f"{project}/{post}"


    if update:
        action = ["add", "--update"]
    else:
        action = ["add"]


    process_list = ['knowledge_repo',
                    '--repo',
                    repo] + action + [filename, '-p', post]
    message = list()
    message.append(subprocess.run(process_list))
    if submit:
        message.append(subprocess.run(["knowledge_repo",
                        "--repo",
                        "repo",
                        "submit",
                        post]))
    if debug:
        return message
    else:
        return
