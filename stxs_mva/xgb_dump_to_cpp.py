import re
import sys

# ========== CONFIGURATION ==========
MODEL_DUMP_FILE = "xgboost_k_fold_3class_model_1.txt"
CPP_OUTPUT_FILE = "xgb_model.cpp"
NUM_CLASSES = 3
# ===================================

def parse_tree(lines, start_idx):
    nodes = {}
    idx = start_idx
    while idx < len(lines):
        line = lines[idx]
        if not line.strip() or re.match(r'booster\[\d+\]:', line):
            break
        # Node line
        m = re.match(r'(\s*)(\d+):\[(.+?)\] yes=(\d+),no=(\d+),missing=(\d+),.*?gain=.*?cover=.*', line)
        if m:
            indent, node_id, cond, yes, no, missing = m.groups()
            feature, thresh = cond.split('<')
            nodes[int(node_id)] = {
                'type': 'split',
                'feature': feature.strip(),
                'thresh': float(thresh),
                'yes': int(yes),
                'no': int(no),
                'missing': int(missing)
            }
        else:
            # Leaf node
            m = re.match(r'(\s*)(\d+):leaf=([-\d\.e]+),.*', line)
            if m:
                indent, node_id, value = m.groups()
                nodes[int(node_id)] = {
                    'type': 'leaf',
                    'value': float(value)
                }
        idx += 1
    return nodes, idx

def parse_model_dump(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    trees = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if re.match(r'booster\[(\d+)\]:', line):
            nodes, next_idx = parse_tree(lines, idx+1)
            trees.append(nodes)
            idx = next_idx
        else:
            idx += 1
    return trees

def feature_index(feature_name):
    # XGBoost uses f0, f1, ...
    m = re.match(r'f(\d+)', feature_name)
    if m:
        return int(m.group(1))
    else:
        raise ValueError(f"Unknown feature name: {feature_name}")

def gen_cpp_node(nodes, node_id, indent='    '):
    node = nodes[node_id]
    if node['type'] == 'leaf':
        return f"{indent}return {node['value']};\n"
    else:
        fidx = feature_index(node['feature'])
        code = f"{indent}if (features[{fidx}] < {node['thresh']}) {{\n"
        code += gen_cpp_node(nodes, node['yes'], indent + '    ')
        code += f"{indent}}} else {{\n"
        code += gen_cpp_node(nodes, node['no'], indent + '    ')
        code += f"{indent}}}\n"
        return code

def gen_cpp_tree_function(tree_idx, nodes):
    func = f"float tree_{tree_idx}(const float* features) {{\n"
    func += gen_cpp_node(nodes, 0)
    func += "}\n"
    return func

def gen_cpp_predict_function(num_classes, trees):
    num_trees_per_class = len(trees) // num_classes
    func = f"""
// XGBoost model prediction function
// features: input array of floats (size = number of features)
// proba: output array of floats (size = {num_classes})
void xgb_predict(const float* features, float* proba) {{
    float scores[{num_classes}] = {{0}};
"""
    for i, tree in enumerate(trees):
        class_idx = i % num_classes
        func += f"    scores[{class_idx}] += tree_{i}(features);\n"
    func += """
    // Softmax
    float max_score = scores[0];
    for (int i = 1; i < %d; ++i) if (scores[i] > max_score) max_score = scores[i];
    float sum = 0.0f;
    for (int i = 0; i < %d; ++i) {{
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }}
    for (int i = 0; i < %d; ++i)
        proba[i] = scores[i] / sum;
}
""" % (num_classes, num_classes, num_classes)
    return func

def main():
    trees = parse_model_dump(MODEL_DUMP_FILE)
    with open(CPP_OUTPUT_FILE, 'w') as f:
        f.write('#include <math.h>\n\n')
        for i, nodes in enumerate(trees):
            f.write(gen_cpp_tree_function(i, nodes))
            f.write('\n')
        f.write(gen_cpp_predict_function(NUM_CLASSES, trees))
    print(f"C++ code written to {CPP_OUTPUT_FILE}")

if __name__ == '__main__':
    main()