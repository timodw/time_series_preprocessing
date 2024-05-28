table = """
Raw	28,61%	7,16%	25,00%	9,73%
Standardized	45,05%	43,80%	59,50%	41,59%
Normalized	30,69%	8,73%	26,11%	11,35%
Peak segmented	26,23%	6,56%	25,00%	8,99%
Moving average smoothing (50)	17,13%	4,22%	25,00%	6,64%
Resampled (50)	28,31%	7,69%	25,76%	10,39%
Resampled (25)	25,55%	10,66%	25,18%	9,18%
PCA (25)	28,61%	7,16%	25,00%	9,73%
FFT filter (10)	27,24%	7,04%	25,68%	9,75%
FFT	82,92%	78,33%	86,01%	77,66%
"""

model = 'DEC (GMM)'


if __name__ == "__main__":
    lines = table.strip().split('\n')
    lines = [l.split('\t') for l in lines]
    for i, l in enumerate(lines):
        method = l[0]
        acc = float(l[1][:-1].replace(',', '.'))
        prec = float(l[2][:-1].replace(',', '.')) 
        recall = float(l[3][:-1].replace(',', '.')) 
        f1 = float(l[4][:-1].replace(',', '.'))
        if i > 0:
            print(f"& \\textbf{{{method}}} & ${acc:.2f}\\%$ & ${prec:.2f}\\%$ & ${recall:.2f}\\%$ & ${f1:.2f}\\%$ \\\\")
        else:
            print(f"\\multirow{{{len(lines)}}}{{*}}{{\\textbf{{{model}}}}} & \\textbf{{{method}}} & ${acc:.2f}\\%$ & ${prec:.2f}\\%$ & ${recall:.2f}\\%$ & ${f1:.2f}\\%$ \\\\")

