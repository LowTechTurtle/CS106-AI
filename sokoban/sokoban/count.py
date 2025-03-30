import ast
"""cach su dung file nay: xoa het cac dong tu trong file log
tru cac dong la list trong python, sau do set filename ben duoi cho phu hop va chay"""  

def read_lists_from_file(filename):
    lengths = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                lst = ast.literal_eval(line.strip())  # Chuyển đổi chuỗi thành danh sách
                if isinstance(lst, list):
                    lengths.append(len(lst))
                else:
                    print(f"Warning: Not a list - {line.strip()}")
            except (SyntaxError, ValueError):
                print(f"Error parsing line: {line.strip()}")
    return lengths

# Example usage
filename = "DFS_log.txt"  # Đổi thành đường dẫn file của bạn
lengths = read_lists_from_file(filename)
for i, length in enumerate(lengths, start=1):
    print(f"List {i}: {length} elements")

