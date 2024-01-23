import os
from jinja2 import Environment, FileSystemLoader

path = "./"
#os.makedirs(path, exist_ok=False)


env = Environment(loader=FileSystemLoader(searchpath=""))
template = env.get_template("./main_TROT.py")
N_SEG = 5

for i in range(N_SEG):
    output = template.render({'N_SEG': N_SEG, 'WHICH_PART': i})
    with open(path + "main_TROT_" + str(N_SEG) + '_' + str(i) + ".py", 'w', encoding="utf-8") as out:
        out.write(output)
        out.close()