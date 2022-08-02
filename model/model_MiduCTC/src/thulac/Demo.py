import model.model_MiduCTC.src.thulac as thulac
thu1 = thulac.thulac(seg_only=True)  #只进行分词，不进行词性标注
# thu1 = thulac.thulac()
#
texts1=thu1.cut("被判处刑法的，还要开除公职。")
texts2=thu1.cut("被判处刑罚的，还要开除公职。")
predict_text=thu1.cut('造成粮食歉收。')

print(texts1)
print(texts2)
print(predict_text)

ws="好困，好晚了"
print(thu1.cut(ws))

arr=['你好','才是','大']
print("".join(arr))