from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class Show_tSNE:

    def __init__(self, xList, yList):
        self.xList = xList
        self.yList = yList

    def display_tSNE(self):
        my_color_palette = ["#96415e",
                            "#d7437b",
                            "#c58e92",
                            "#d44851",
                            "#df4028",
                            "#904634",
                            "#c15a2e",
                            "#45372e",
                            "#c09470",
                            "#dda661",
                            "#906b21",
                            "#d9a42e",
                            "#c3c367",
                            "#dbe627",
                            "#c2c943",
                            "#566c2b",
                            "#b6bfa2",
                            "#95db39",
                            "#79b24d",
                            "#a3cf93",
                            "#61d64f",
                            "#486a45",
                            "#5dd482",
                            "#5fd1b8",
                            "#70a3a2",
                            "#63adcd",
                            "#7c94d2",
                            "#6978e0",
                            "#4d4b8c",
                            "#382d4d",
                            "#6a40cd",
                            "#4d258c",
                            "#a39aaa",
                            "#bb53e6",
                            "#bd7cd2",
                            "#b992bc",
                            "#832f78",
                            "#d644ba",
                            "#d771af"]

        different_64_colors = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
                               "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
                               "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
                               "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
                               "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
                               "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
                               "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
                               "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
                               "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
                               "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
                               "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
                               "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
                               "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"]
        marker = ['+', 'o', '^']  # source domain, labeled target domain, unlabeled target domain

        tsne = TSNE(n_components=2)
        source_z = tsne.fit_transform(self.xList)

        tsne = TSNE(n_components=2)
        taregt_z = tsne.fit_transform(self.yList)

        fig, ax = plt.subplots()

        for aSample in source_z:
            ax.plot(aSample[0], aSample[1], marker='o', markersize=3, color="red")
        for i in range(source_z.shape[0]):
            plt.text(source_z[i, 0], source_z[i, 1], str(i),color="red")


        for aSample in taregt_z:
            ax.plot(aSample[0], aSample[1], marker='o', markersize=3, color="blue")
        for i in range(taregt_z.shape[0]):
            plt.text(taregt_z[i, 0], taregt_z[i, 1], str(i),color="blue")
        plt.show()
        print()

