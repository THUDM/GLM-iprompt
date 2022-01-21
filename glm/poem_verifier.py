

import jsonlines
def cilin():
    f=open("cilin.txt",'r')
    t=f.readlines()
    bu=0
    nowsb=0
    allbu=[]
    def_chr=['[',']','(',')','\n',' ','，','\u3000','。','《','》']
    allsb=[[],[]]
    worddict={}
    shengdict={}
    for line in t:
        if len(line)<5:
            continue
        if ('第' in line) and ('部' in line):
            bu+=1
            allbu.append([])
        if ('平声' in line):
            nowsb=0
        if ('仄声' in line):
            nowsb=1
        if ('入声' in line):
            nowsb=1
            
        if ('【' in line):
            tks=line.split('】')[1]
            currentst1=0
            currentst2=0
            for num in range(len(tks)):
                char=tks[num]
                if currentst1+currentst2==0:
                    if not(char in def_chr):
                        allbu[-1].append(char)
                        allsb[nowsb].append(char)
                        if char in worddict:
                            if not(bu in worddict[char]):
                                worddict[char].append(bu)
                        else:
                            worddict[char]=[bu]
                        if char in shengdict:
                            if not(nowsb in shengdict[char]):
                                shengdict[char].append(nowsb)
                        else:
                            
                            shengdict[char]=[nowsb]
                            
                if char=='[':
                    currentst1=1
                if char==']':
                    currentst1=0
                if char=='(':
                    currentst2=1
                if char==')':
                    currentst2=0
    
    
    return worddict,shengdict,allbu,allsb
    #print(allbu[0])

def get_last_piece(sentence):
    i=-2
   
    while (-i<len(sentence)) and not(sentence[i] in [',','，','.','。',':','：','?','？','!','！',';','；','>']):
        i-=1
    #print(sentence,sentence[i+1:])
    return sentence[i+1:]

def eng_verifier(sentence,verifier_params):
    tokenizer=verifier_params[0]
    decode_tokens = tokenizer.DecodeIds(sentence.cpu().tolist())
    if 'Question' in decode_tokens[10:]:
        return -1000
    if '…' in decode_tokens:
        return -1000
    return 0
    
def code_verifier(sentence,verifier_params):
    tokenizer=verifier_params[0]
    decode_tokens = tokenizer.DecodeIds(sentence.cpu().tolist())
    if '<|end' in decode_tokens:
        return -1,decode_tokens.split('<|end')[0]
    lines=decode_tokens.split('\n')
    remaining_st=[]
    count=0
    for i in range(len(lines)):
        line=lines[i]
        
        if len(line)>0:
            if line[0]!=' ':
                count+=1
                if count>1:
                    return -1,remaining_st
        remaining_st=remaining_st+line+'\n'
        
    return 0,sentence
    
def poem_verifier(sentence,verifier_params,print_reason=False):
    
    tokenizer,wdic,shengdict,rhy,endrhy,min_length,max_length,end_tokens,yayun=verifier_params
   
    decode_tokens = tokenizer.DecodeIds(sentence.cpu().tolist())
    decode_token=get_last_piece(decode_tokens)
    icount=0
    
    if len(decode_token)==0:
        if print_reason:
            print(st,'异常')
        return -1000
        
    prev=decode_tokens[:-len(decode_token)]
    st=decode_token
    for i in range(len(decode_token)-2):
        if decode_token[i:i+3] in prev:
            if print_reason:
                print(st,'重复')
            return -1000
   
    
    
    fullst=False
    
    if (len(st)>0 and (st[-1] in end_tokens)):
        st=st[:-1]
        fullst=True
        if len(st)<min_length:
            if print_reason:
                print(st,'过短')
            return -1000
        if len(st)==6:
            if print_reason:
                print(st,'字数不合')
            return -1000
        if decode_token[len(st)-1:] in prev:
            if print_reason:
                print(st,'重复')
            return -1000
    
    if len(st)>max_length:
        if print_reason:
            print(st,decode_tokens,end_tokens,'过长')
        return -1000
    if len(st)==max_length:
        fullst=True
        if st[-1]+'。' in prev:
            if print_reason:
                print(st,'重复')
            return -1000
        if st[-1]+'，' in prev:
            if print_reason:
                print(st,'重复')
            return -1000
    for i in range(len(st)):
        if st[i] in prev:
            icount-=3
        if st[i] in st[:i]:
            icount-=3
        if i!=len(st)-1:
            if st[i:i+2] in prev:
                icount-=7.5
            if st[i:i+2] in st[:i]:
                icount-=7.5
        if st[i] in ['的','些','么','了']:
            icount-=5
                
                
    for i in st:
        if not(i in shengdict):
            if print_reason:
                print(st,i)
            return -1000
            
    if len(st)<2:
        return icount
    
    
    pz1=shengdict[st[1]]
    if rhy!=2:
        if len(pz1)==1:
            if rhy!=pz1[0]:
                if print_reason:
                    print(st,'第2字平仄')
                return -1000
    if rhy==2:
        if len(pz1)==1:
            rhy=pz1[0]
        
    if len(st)>=4:
        pz2=shengdict[st[3]]
        if rhy!=2:
            if len(pz2)==1:
                if pz2[0]+rhy!=1:
                    if print_reason:
                        print(st,'第4字平仄')
                    return -1000
        if rhy==2:
            if len(pz2)==1:
                rhy=1-pz2[0]
    #rhy: 0: 010  1:101  2: not decided
    #  endrhy:0/1, 2:undecided
  
    if len(st)>max_length-3:
    #regulate the 3rd word
        if endrhy!=2:
            
            wrhy=rhy
            if max_length==5:
                wrhy=1-rhy
            if endrhy==wrhy:
                pz=shengdict[st[max_length-3]]
                if len(pz)==1:
                    if pz[0]==wrhy:
                        if print_reason:
                            print(st,'三连')
                        return -1000
                        
    if len(st)>=6:
        
        pz3=shengdict[st[5]]
        if rhy!=2:
            if len(pz3)==1:
                if rhy!=pz3[0]:
                    if print_reason:
                        print(st,'第6字平仄')
                    return -1000
    
    if fullst:
            
        pz11=shengdict[st[-3]]
        pz12=shengdict[st[-2]]
        pz13=shengdict[st[-1]]
        if len(pz11)+len(pz12)+len(pz13)==3:
            if pz11[0]+pz12[0]+pz13[0]==0:
                if print_reason:
                    print(st,'三连平')
                return -1000
            if pz11[0]+pz12[0]+pz13[0]==3:
                if print_reason:
                    print(st,'三连仄')
                return -1000
    
        if endrhy!=2:
            if len(pz13)==1:
                
                if endrhy!=pz13[0]:
                    if print_reason:
                        print(st,'仄起平收')
                    return -1000
        if endrhy==2:
            if len(pz13)==1:
                endrhy=pz13[0]
            
        if (len(yayun)>0 and endrhy==0):
            final1=wdic[st[-1]]
            final2=[]
            for i in yayun:
                final2.append(wdic[i])

            doc=0
            for i in final1:
                doc=1
                for td in final2:
                    if not(i in td):
                        doc=0
                if doc==1:
                    break
            if doc==0:
                if print_reason:
                    print(st,'押韵')
                return -1000
            
    return icount
        
    
def verify_rhy(sentence,id,shengdict,yayun,rhy):
    st=get_last_piece(sentence)
    
    st=st[:-1]
    length=len(st)
    pz=shengdict[st[-1]]
    needyy=0
    if (id==0):
        if len(pz)==1:
            if pz[0]==0:
                needyy=1
    if id%2==1:
        needyy=1
    if needyy==1:
        yayun=yayun+st[-1]
    rhy=2
    pz1=shengdict[st[1]]
    if len(pz1)==1:
        rhy=pz1[0]
        return yayun,rhy,length
    pz2=shengdict[st[3]]
    if len(pz2)==1:
        rhy=1-pz2[0]
        return yayun,rhy,length
    if len(st)>5:
        pz3=shengdict[st[5]]
        if len(pz3)==1:
            rhy=pz3[0]
        
    return yayun,rhy,length
    
def get_pron():
    file='cipai.txt'
    f=open(file,'r')
    lines=f.readlines()
    cp={}
    alllist=[]
    for line in lines:
        linsp=line.split(':')
        if len(linsp)>1:
        #shuangdiao
            cp[linsp[0]]=linsp[1].replace('\n','')
            alllist.append(linsp[0])
    return cp,alllist
