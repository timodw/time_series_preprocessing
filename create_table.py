table = """
Raw	13,43%	33,02%	37,12%	19,90%
Standardized	12,15%	32,51%	36,07%	18,78%
Normalized	11,66%	32,13%	35,74%	18,15%
Peak segmented	13,85%	33,41%	37,43%	20,27%
Moving average smoothing (50)	39,64%	49,89%	57,32%	41,65%
Resampled (50)	19,98%	37,61%	40,10%	24,86%
Resampled (25)	20,71%	42,42%	42,09%	27,15%
PCA (25)	17,01%	36,04%	39,64%	23,45%
FFT filter (10)	14,05%	40,79%	37,70%	20,39%
FFT	44,36%	51,83%	67,67%	49,93%
"""

learning_method = 'Unsupervised'
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
            print(f"& & \\textbf{{{method}}} & ${acc:.2f}\\%$ & ${prec:.2f}\\%$ & ${recall:.2f}\\%$ & ${f1:.2f}\\%$ \\\\")
        else:
            print(f"\\multirow{{{len(lines)}}}{{*}}{{\\textbf{{{learning_method}}}}} & \\multirow{{{len(lines)}}}{{*}}{{\\textbf{{{model}}}}} & \\textbf{{{method}}} & ${acc:.2f}\\%$ & ${prec:.2f}\\%$ & ${recall:.2f}\\%$ & ${f1:.2f}\\%$ \\\\")

