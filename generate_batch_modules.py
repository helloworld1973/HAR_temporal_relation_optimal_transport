import os
from jinja2 import Environment, FileSystemLoader

path = "./"
#os.makedirs(path, exist_ok=False)


env = Environment(loader=FileSystemLoader(searchpath=""))
template = env.get_template("./main.py")

adaptationAlgoUsed = ["NA", "Tused", "LMNN", "SA", "CORAL", "OT", "OTDA", "MLOT", "SOT", "JDOT", "TCA"] # LMNN OTDA

for i in range(0, len(adaptationAlgoUsed)):
    output = template.render({'algo_name': adaptationAlgoUsed[i]})
    with open(path + "main_" + adaptationAlgoUsed[i] + ".py", 'w', encoding="utf-8") as out:
        out.write(output)
        out.close()