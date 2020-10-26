def generate_layers_id(layer_num, keep_num):
    binary_codes_list = []

    def dfs(cur_ind, code):
        if cur_ind == layer_num-1:
            binary_codes_list.append(code)
            return 

        dfs(cur_ind+1, code+"0")
        dfs(cur_ind+1, code+"1")
        return 

    dfs(-1, "")
    
    res = [[i for i in range(layer_num) if binary_code[i] == '1'] for binary_code in binary_codes_list]
    res = list(filter(lambda x: len(x)==keep_num, res))

    return res, binary_codes_list


def convert_list_to_string(li):
    """
    example:
    input: [0, 1, 3, 5] 
    output: 0,1,3,5
    """
    return ",".join(map(str, li))



if __name__ == "__main__":
    res = generate_layers_id(6, 4)
    print(res)


        

