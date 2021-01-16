from graphviz import Digraph
from IPython.display import Image

def create_digraph(name, nodes, edges, filename):
    g = Digraph(name, filename=filename, format='png', engine='dot')
    g.attr(size='8')
    g.attr('node', shape='oval', style='filled', color='lightgrey',)
    for n in nodes:
        g.node(**n)

    for e in edges:
        g.edge(e['start'], e['end'], **e)


    g.render()
    Image(filename+'.png')
    return g
