class lz78():
    ############ ENCODING ###########
    def encoding(self, text):
        """Compress a string to a list of output symbols."""
        l_c = list(text)
        d = dict()
        count = 1
        i = 0
        result = []
        for idx in range(len(l_c)):
            w = l_c[i]
            if w not in d.values() or i==(len(l_c)-1):
                d[count] = w
                count +=1
                i += 1
                result.append([0,w])
            else:
                i += 1
                w = w + l_c[i]
                while(w in d.values() and (i==(len(l_c)))):
                    i += 1
                    w = w + l_c[i]
                d[count] = w
                count +=1
                key = list(d.keys())[list(d.values()).index(w[:-1])]
                result.append([key,w[-1]])
                i += 1
            if i>=len(l_c):
                break
        return result
    
    ########## DECODING ############
    def decoding(self, lista_chaves):
        result = ""
        d = dict()
        count = 1
        for key, val in lista_chaves:
            if key == 0:
                d[count] = val
                count += 1
                result += val
            else:
                w = d[key] + val
                result += w
                d[count] = w
                count += 1
        return result
    
obj = lz78()
text = "abbbabbbrcaaarcab"
l_e = obj.encoding(text)
print(l_e)
print(obj.decoding(l_e))