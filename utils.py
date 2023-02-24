"""
Takes an array of labels, e.g. ["Male","Female", ...], and convert it to a csv with the string 0,1,...
"""
def gen_csv_from_pred(predictions, model_name):
    str=""
    for i in range(len(predictions)):
        comma = "," if i != len(predictions)-1 else ""
        label = int(predictions[i].lower() == 'female')
        str += f"{label}{comma}"

    with open(f'data/{model_name}_predictions.csv', 'w') as f:
        f.write(str)
