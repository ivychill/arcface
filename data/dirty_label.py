#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0.0.01
@author: qingyao.wang
@license: kuang-chi Licence 
@contact: qingyao.wang@kuang-chi.com
@site: 
@software: PyCharm
@file: statistical_2.py
@time: 2019/11/16 15:11
"""
import os,shutil

orgPath = 'D:/wqy/new_dataset/cam'
ID_Path = 'D:/wqy/new_dataset/id_2/id'
TF_Path = 'D:/wqy/allimportant_42w'
out_root = 'D:/wqy/误识/outface'



file = open('out_result_42w_6800.txt')
retlist=[]
for line in file:
    retlist.append(line)
file.close()

def reSaveImg(orgPID, orgImg,_PID,_IMG,sender,threshold,saveType):
    campath = os.path.join(os.path.join(orgPath,orgPID),orgImg+'.jpg')
    if saveType=='unrecall':  #未召回
        targpath = '{}/{}/{}/{}'.format( out_root, sender,str(threshold),saveType)
        if not os.path.exists(targpath):
            os.makedirs(targpath)
        targfile = '{}/{}_{}.jpg'.format(targpath,orgPID,orgImg)
        shutil.copyfile(campath,targfile)

    if saveType=='error': # 误识别
        filename1 = '{}_{}_{}_{}_1.jpg'.format(orgPID, orgImg,_PID,  _IMG)
        filename2 = '{}_{}_{}_{}_2.jpg'.format(orgPID, orgImg, _PID, _IMG)
        if _PID == _IMG:
            srcpath = os.path.join(TF_Path,_PID + '.jpg')
            targpath = '{}/{}/{}/{}'.format(out_root, sender,str(threshold),  saveType )
            if not os.path.exists(targpath):
                os.makedirs(targpath)
            targfile1 = '{}/{}'.format(targpath,filename1)
            targfile2 = '{}/{}'.format(targpath, filename2)

            shutil.copyfile(campath, targfile1)
            shutil.copyfile(srcpath, targfile2)
        else:
            srcpath = '{}/{}/{}.jpg'.format(ID_Path, _PID ,_IMG )
            targpath = '{}/{}/{}/{}'.format(out_root, sender,str(threshold),  saveType)
            if not os.path.exists(targpath):
                os.makedirs(targpath)

            targfile1 = '{}/{}'.format(targpath, filename1)
            targfile2 = '{}/{}'.format(targpath, filename2)

            shutil.copyfile(campath, targfile1)
            shutil.copyfile(srcpath, targfile2)

def special_handling(orgPID,cpy_PID):
    #return False
    if (orgPID == '23102219721025541X' and cpy_PID == '231022197210255429') or \
            (orgPID == '231022197210255429' and cpy_PID == '23102219721025541X') or \
            (orgPID == '31022219640615363X' and cpy_PID == 'E1A87DD16374490C9FC230F0ED7B4260') or \
            (orgPID == '31010819681210411X' and cpy_PID == '0B06DE642DEE4CEF821E5B6983C753D1') or \
            (orgPID == '31022419690127102X' and cpy_PID == '112F19F36E754D5580C2F663A4E804A9') or \
            (orgPID == '34012219720311001X' and cpy_PID == '340122197203110033') or \
            (orgPID == '35040219780714001X' and cpy_PID == '70DA815EA2A042F4A8FA8383285DFF4F') or \
            (orgPID == '130703197212160347' and cpy_PID == 'D3B8EEE8516C465B9696F4B6F1332C90') or \
            (orgPID == '140423199012120076' and cpy_PID == '3159481C69424ACEB1B09F46E8A1A290') or \
            (orgPID == '142223197109132753' and cpy_PID == 'F7F042F22C8A42828134AF707C0F1E0B') or \
            (orgPID == '142431199104290023' and cpy_PID == '142431199104290031') or \
            (orgPID == '142431199104290031' and cpy_PID == '142431199104290023') or \
            (orgPID == '32070319790607051X' and cpy_PID == '7971F6B0EDCA438D890EBFC4E65F3631') or \
            (orgPID == '142729197006040616' and cpy_PID == 'AADCBC0AAC3E4300966C494D97AE7FB5') or \
            (orgPID == '310101198111041018' and cpy_PID == '66B6DCC5B1FF4EDBA1E0E4ABC19EEFAC') or \
            (orgPID == '310101195907164079' and cpy_PID == 'BF8FC45CE1524AE1BCD4E4412C3055A7') or \
            (orgPID == '310101196103290015' and cpy_PID == 'DB408E366BE8490C8BE75DE24770F46C') or \
            (orgPID == '310101196901150818' and cpy_PID == 'C068FAA35365428FB286D8F9723E999C') or \
            (orgPID == '310102196209091272' and cpy_PID == '7B0F8E738EE44D66B63956ECDE9222A9') or \
            (orgPID == '310102198710101218' and cpy_PID == 'C38F0D7634EC4EEE9453CE0986D96FF8') or \
            (orgPID == '310103195309042031' and cpy_PID == 'E11D223543EC46E4A4CBA677A6F9B3BA') or \
            (orgPID == '310103195508174483' and cpy_PID == 'CFE282316C44411DBD556571F64217F6') or \
            (orgPID == '310103196011131618' and cpy_PID == '8B82CD2BA593489AA2236025C2BDD39E') or \
            (orgPID == '310103196605130073' and cpy_PID == '8E86351CAEF24E70B169C65570AE8E1F') or \
            (orgPID == '310103196805073210' and cpy_PID == '2D5F219693884C309637EDFAD9107266') or \
            (orgPID == '310104197501030810' and cpy_PID == 'F2F4DEC085FA4D65930FE88D06056710') or \
            (orgPID == '310104198402267219' and cpy_PID == '9D677B787C6C48AEB4DC92D6DF7D43FC') or \
            (orgPID == '310105195810290437' and cpy_PID == 'EF45E6A434294E1EB7B4B89105A48026') or \
            (orgPID == '310105196907170028' and cpy_PID == '310105196907170030') or \
            (orgPID == '310105196907170030' and cpy_PID == '310105196907170028') or \
            (orgPID == '310107195904274019' and cpy_PID == 'AC5BC5D831DB4871889358446C240F2B') or \
            (orgPID == '310107196007083225' and cpy_PID == '6198D675DD784E89B509E81F71B45A40') or \
            (orgPID == '310107197809025419' and cpy_PID == 'E114C4A2A9F74065803C358437A243F9') or \
            (orgPID == '310107198106271319' and cpy_PID == '56D4499E669E462399101C117AE08E59') or \
            (orgPID == '310107198203263919' and cpy_PID == '410524971620444DBAE780DAF4B6C07A') or \
            (orgPID == '310108196309290429' and cpy_PID == '1A6120B49C424946B4DF072DFA23C675') or \
            (orgPID == '310108197211263216' and cpy_PID == '310108197211263236') or \
            (orgPID == '310108197211263236' and cpy_PID == '310108197211263216') or \
            (orgPID == '310108197506224011' and cpy_PID == '1B033C528DF048E19D63A4E06FFA8F8A') or \
            (orgPID == '310110195801235058' and cpy_PID == 'E9B8A8789148410EAC423A981C49FE29') or \
            (orgPID == '310114198605171025' and cpy_PID == '83CBC03C90DD4157A6E09A986D20CB7A') or \
            (orgPID == '310221195201203612' and cpy_PID == 'D277121258C0425D80C71BED134D57E1') or \
            (orgPID == '310222196002283673' and cpy_PID == '756AEBCB418A4D2A82D4C0F7C213E6D5') or \
            (orgPID == '310222196002283673' and cpy_PID == 'BACF093804284BD5A7AD64E121993E3E') or \
            (orgPID == '310225196208100819' and cpy_PID == '7BBA2EE09E424B8E9F52E035B1D6FAAC') or \
            (orgPID == '310228196902022039' and cpy_PID == '48EFEEB46AB54A76B723ECA847730485') or \
            (orgPID == '310222194302023663' and cpy_PID == 'F979FF0C90094FB0BE64E5E8B5D991D8') or \
            (orgPID == '310228198212283417' and cpy_PID == '21784543165C47A7A32D40ADA55E7FC7') or \
            (orgPID == '310229194608201615' and cpy_PID == '86625ED66A874A38AD0ADA9DDFDF2709') or \
            (orgPID == '310230194510267224' and cpy_PID == '06A22F2B25754B3AA687FA76E76169E8') or \
            (orgPID == '310230195606307467' and cpy_PID == '09D582CC635A4124A860E03982033482') or \
            (orgPID == '310230195606307467' and cpy_PID == '525AB5343F77494AADA33A9B5A9F9E41') or \
            (orgPID == '340323198205130016' and cpy_PID == '8E41E3D9CD6642058B87884D41C09C63') or \
            (orgPID == '342623197805308117' and cpy_PID == '05D23FA74F064A74943FD113D81825ED') or \
            (orgPID == '342122197410157576' and cpy_PID == 'AED5B8A5AA2A4C98BBF27DC19DDC8C4C') or \
            (orgPID == '350424198808112013' and cpy_PID == '40FBCB2B7A7F49A5A894B38C592AA690') or \
            (orgPID == '350524199011148318' and cpy_PID == 'BD3A91DFC9A64CFC8CBD4BB5F353D50D') or \
            (orgPID == '362202197601151533' and cpy_PID == 'D744262F70F04CD6BC2A26F351CC0BF0') or \
            (orgPID == '370403197907132711' and cpy_PID == '092CF62ED71F477C8B0699B68F6069CD') or \
            (orgPID == '371121198412250013' and cpy_PID == '371121198412250025') or \
            (orgPID == '371121198412250025' and cpy_PID == '371121198412250013') or \
            (orgPID == '413023197002142017' and cpy_PID == 'E7B8FD89A9E34BF99D9060C5CACBA791') or \
            (orgPID == '421124199507242516' and cpy_PID == 'AA97F3E8359E49598152ABE0ADEEE52E') or \
            (orgPID == '445221197812035323' and cpy_PID == 'E6F02FC9C62D4C24B0C99F0B83A81F40') or \
            (orgPID == '452421198103021433' and cpy_PID == 'BDF340B24A4A473BABBB928F8B037E5B') or \
            (orgPID == '452527197911162348' and cpy_PID == '2865EB98A8104B258E6E1CBA61BC7C93') or \
            (orgPID == '612522196609162631' and cpy_PID == 'FA5C82AE69BC4CC98819C645C7257423') or \
            (orgPID == '632123199606088423' and cpy_PID == '2DBB22836EC8426AB91AEA24F118BC26') or \
            (orgPID == '522425198709068723' and cpy_PID == '14C5901C3FC04FB7AB20CA0B9096439F') or \
            (orgPID == '513701199010011757' and cpy_PID == 'FD55C0E0C1524B30A96A2E3398536F55') or \
            (orgPID == '31022219640615363X' and cpy_PID == '76066A8036184FDC847E49F257521A56') or \
            (orgPID == '31022819700114501X' and cpy_PID == 'D6A7977A7A594B028CA59604010F8C70') or \
            (orgPID == '310105195810290437' and cpy_PID == 'E59DF46969FB4AB3AE5BFB499BAD2572') or \
            (orgPID == '310108195208034069' and cpy_PID == 'D8BC6208588046D682401BB439139B46') or \
            (orgPID == '310110197603256227' and cpy_PID == '310110197603256244') or \
            (orgPID == '310110197603256244' and cpy_PID == '310110197603256227') or \
            (orgPID == '310110197411234630' and cpy_PID == '409686FA84FF4722ACA01C9015DAD10F') or \
            (orgPID == '310228197202233833' and cpy_PID == '2DEE0D7F77D141F7A21526A131AD35DD') or \
            (orgPID == '432923196311033022' and cpy_PID == '432923197208148141') or \
            (orgPID == '432923197208148141' and cpy_PID == '432923196311033022') or \
            (orgPID == '430481197311013936' and cpy_PID == '75C1D1C55A3749DDB3137912D067EBC0') or \
            (orgPID == '310103196011131618' and cpy_PID == 'EFD004769E924AA2A9DBBCB20B6D186C') or \
            (orgPID == '310103196011131618' and cpy_PID == '64B1C51458CF4CABB2563BE50407A5F6') or \
            (orgPID == '31022419690127102X' and cpy_PID == 'D80F9FC9C01343CC88BC5CC157D20169') or \
            (orgPID == '210102198401135031' and cpy_PID == '813CD51C1C8D426ABBC1FF038B7630D6') or \
            (orgPID == '513231197210040511' and cpy_PID == 'F0FC40F33E3E42009840FCC1601B6993') or \
            (orgPID == '452724197910122530' and cpy_PID == '8973798D0199497FAE496BB1CCB307B1') or \
            (orgPID == '310222196002283673' and cpy_PID == 'F2377F09DA5849E5B656C69DC6A8BDA0') or \
            (orgPID == '310222196002283673' and cpy_PID == '82AD49A977C54FA5951DD0A33CE5C9D1') or \
            (orgPID == '310222194302023663' and cpy_PID == '708DC06B4D5E4288BDF1CF303FD6E631') or \
            (orgPID == '310115198507030620' and cpy_PID == '0A919861ADA6419B9239836CDDA1AB55') or \
            (orgPID == '310108196011191241' and cpy_PID == '83D4C4E145664DDBBE6DF5D7EB865EB9') or \
            (orgPID == '310109196202202835' and cpy_PID == '8D153718EF1C41899E864982B4393A3B') or \
            (orgPID == '310109198611123061' and cpy_PID == '61E6C588434A42CFAF3BA6CD06A8D5B9') or \
            (orgPID == '310230194510267224' and cpy_PID == '800F6A99254F489BB9FFDA0CD61F5B9E') or \
            (orgPID == '310224195604127035' and cpy_PID == 'B67C03C6155540938AF98FF37D339B5F') or \
            (orgPID == '310224195604127035' and cpy_PID == 'BB15DA6B900A4AC3B15EAF26354210F6') or \
            (orgPID == '310115197811121717' and cpy_PID == '0CA3AF88973F492E81645F7F054BA828') or \
            (orgPID == '310230197902242951' and cpy_PID == '8F85D33F87B643AF8DC36ADE56A0EB3B') or \
            (orgPID == '310229195608271626' and cpy_PID == 'A344F62FFDD34B748374083315FEA0CD') or \
            (orgPID == '340122197203110033' and cpy_PID == '34012219720311001X') or \
            (orgPID == '34012219720311001X' and cpy_PID == '340122197203110033') or \
            (orgPID == '340122198311197372' and cpy_PID == '7D21464FEB924DFCA3B38966512637A5') or \
            (orgPID == '35040219780714001X' and cpy_PID == '0592233A3FE44997A997F7388F0ADCCE') or \
            (orgPID == '230102199004172419' and cpy_PID == '04A2E9C8C11A42A2820219795E5C86C3') or \
            (orgPID == '432925194711116920' and cpy_PID == '7A4DD2CC977E4B43807AFF11B33F4D14') or \
            (orgPID == '653126199504150859' and cpy_PID == '839C6ADB1E12477AA2AB65D3C9460081') or \
            (orgPID == '310110196007285052' and cpy_PID == '8D5FE8612EA84E9AB8E2C1AD6BA86EBB') or \
            (orgPID == '310109196812172012' and cpy_PID == 'E1062C4F3D6F480EBF798D07BB0E7B3A') or \
            (orgPID == '310110195703015510' and cpy_PID == '528EF96A8EC84C13A82A8DD54EE9B942') or \
            (orgPID == '310114198605171025' and cpy_PID == 'F161F5C616A44661AEF4F93A2B3F8691') or \
            (orgPID == '310222193410113610' and cpy_PID == 'EE9A5190A12E49AE9675D20C166E2DB5') or \
            (orgPID == '310115198507030620' and cpy_PID == '2851EAB9BD56477A9D8A73FC685B68B6') or \
            (orgPID == '310229197808014818' and cpy_PID == '64ED9369EDFC4D368EF5934E09699BBC') or \
            (orgPID == '341226199202172615' and cpy_PID == 'D1C874B9E1C847C0BF4F831A6144F46A') or \
            (orgPID == '341122198410204217' and cpy_PID == '586FE8923B7445F3927DF3D2CA03FA95') or \
            (orgPID == '340122198311197372' and cpy_PID == '07E4897E5556487AA6A34CD807ED9EDC') or \
            (orgPID == '372923198506065975' and cpy_PID == '372923198708143812') or \
            (orgPID == '372923198708143812' and cpy_PID == '372923198506065975') or \
            (orgPID == '31022419690127102X' and cpy_PID == '0E7EDFE894934B2AAC4F249277B60182') or \
            (orgPID == '362429195209280025' and cpy_PID == 'FA97AA9C812D43BA8AC2D3AC431344BD') or \
            (orgPID == '421127198502085210' and cpy_PID == '48B115E7AE064D79BA09C23396EC177B') or \
            (orgPID == '310105198609101757' and cpy_PID == '37EDA721076A408B8C53524B4A976DF2') or \
            (orgPID == '310108195208034069' and cpy_PID == 'BECFE7DB66034ACBB53F24DFA6B00C7A') or \
            (orgPID == '310102196209091272' and cpy_PID == '96A8467C6D9345F6BDFC20BCB4ECF5D3') or \
            (orgPID == '310109196006282436' and cpy_PID == '628990F98BDB4CFE9E40ECF844C01022') or \
            (orgPID == '310229195201261610' and cpy_PID == '0FF50EE9FC914C2B9EF4ADE96963C794') or \
            (orgPID == '320311194903310420' and cpy_PID == '62E636ABF7464FAEB1D58D06230662E7') or \
            (orgPID == '310108196011191241' and cpy_PID == '4C654D1A9BA641A2B07138827E75DDDE') or \
            (orgPID == '310108196309290429' and cpy_PID == '2D8E3099583741E28959B99A7BC51384') or \
            (orgPID == '310105195810290437' and cpy_PID == 'A1464847BCB84703BC536F7084B2376D') or \
            (orgPID == '31022219640615363X' and cpy_PID == '7DB9AE40AA014E979447D192766B7026') or \
            (orgPID == '310102195802053619' and cpy_PID == 'BD5004898DAE463A89B23AB52C14CE82') or \
            (orgPID == '320223198107286173' and cpy_PID == 'E8254F3336FE486DA2458C6DC0B6CD60') or \
            (orgPID == '31010519920825161X' and cpy_PID == '606C09A83B1546C9AB07943B082C0625') or \
            (orgPID == '31022819790820361X' and cpy_PID == '8FD154CE716840B3BB5AFE80CE8F8BD7') or \
            (orgPID == '35260219780709161X' and cpy_PID == '849002298AA9444A888BB88436774047') or \
            (orgPID == '37083219820521493X' and cpy_PID == '5814C05F1B6A4411ACCCB4D25D67BD84') or \
            (orgPID == '130481197912082717' and cpy_PID == '51F2A2A772F1465D9BCD64F5440D9B9E') or \
            (orgPID == '211224197506262758' and cpy_PID == 'CFCDF16C48514C8AA4BC722CB48CE4ED') or \
            (orgPID == '220202198511287228' and cpy_PID == '541CD98EB2C04FDEB7756F3471A43CB9') or \
            (orgPID == '230223198706081610' and cpy_PID == '1AFBD5F6B35048B0ABB00A877A7D110A') or \
            (orgPID == '230823197401180018' and cpy_PID == 'DC84FAF0C6B8424EA720DA02F2E1B776') or \
            (orgPID == '310102195802053619' and cpy_PID == 'BD5004898DAE463A89B23AB52C14CE82') or \
            (orgPID == '310106195111081616' and cpy_PID == '5635CCD3CD9C471DAF729DE119B7888D') or \
            (orgPID == '310108195602103237' and cpy_PID == '369836D6D105466ABE81159DF2DD818F') or \
            (orgPID == '310109194502194815' and cpy_PID == '12EF830CF6864537B858238AC1B94A08') or \
            (orgPID == '310109196511221255' and cpy_PID == '3107B818A95D41808BD093B75AC75567') or \
            (orgPID == '310109198202111537' and cpy_PID == 'F35F2FD4596F46D4841D5EA3D3E4F7ED') or \
            (orgPID == '310110196007285052' and cpy_PID == '345800C17372462D9D8666CC6B47A858') or \
            (orgPID == '310110196012243244' and cpy_PID == '46828B0EF3224915B853AF157C7157D5') or \
            (orgPID == '310110198312035614' and cpy_PID == '6E0BFF9E70534EDDBCBD2FF1E2BB6C81') or \
            (orgPID == '310115198701257714' and cpy_PID == 'C32A14FCBAA24DBEA681B8E218898B52') or \
            (orgPID == '310221195201203612' and cpy_PID == 'F137D7E607D84F54B0370D81664C9758') or \
            (orgPID == '310222194307203612' and cpy_PID == 'D5E1CE6EBAB741DC90FE881451BD1BC5') or \
            (orgPID == '310222194307203612' and cpy_PID == '1CC2C319F4114D6BB7A758AD1D0CBB60') or \
            (orgPID == '310222194402273811' and cpy_PID == '9C04DF150DD6441384FF3503CCF10DB0') or \
            (orgPID == '310228196208063615' and cpy_PID == 'EEE48904C8A2448CB4E782ACB7469CBD') or \
            (orgPID == '310230196301221499' and cpy_PID == '3FF5B6E6A0374FD693DE220104947AE2') or \
            (orgPID == '320223198107286173' and cpy_PID == 'E8254F3336FE486DA2458C6DC0B6CD60') or \
            (orgPID == '320822197210222137' and cpy_PID == '8632A1E0CAC0423A8F7C09B63B6171B3') or \
            (orgPID == '321084199207243215' and cpy_PID == '310109198102152016') or \
            (orgPID == '310109198102152016' and cpy_PID == '321084199207243215') or \
            (orgPID == '330211197004250719' and cpy_PID == 'C7E68AEE1A004A9988553274B5F85FFD') or \
            (orgPID == '330902198411260312' and cpy_PID == '1EED130D77BF41D48688A92989EBF145') or \
            (orgPID == '340621197511065258' and cpy_PID == '1D75A554984742698AE3F05F33D7C3EB') or \
            (orgPID == '342426197107170016' and cpy_PID == '5087C8C111A94CA9A74F3E80131B1A37') or \
            (orgPID == '350125196501012916' and cpy_PID == '277EE512DF61406899164A53E211F93E') or \
            (orgPID == '350782197201204515' and cpy_PID == '1AD59D1A6D914E34BC29FE58FAF18940') or \
            (orgPID == '352230198105090629' and cpy_PID == 'F77620A13762434191C8046906108BC6') or \
            (orgPID == '371312198601016474' and cpy_PID == '310230198512242512') or \
            (orgPID == '310230198512242512' and cpy_PID == '371312198601016474') or \
            (orgPID == '410204196408274014' and cpy_PID == '310107196808284617') or \
            (orgPID == '310107196808284617' and cpy_PID == '410204196408274014') or \
            (orgPID == '412322197705157812' and cpy_PID == '371525198409111714') or \
            (orgPID == '371525198409111714' and cpy_PID == '412322197705157812') or \
            (orgPID == '412724198008206112' and cpy_PID == '992844447DC6417DA222988FA23DD842') or \
            (orgPID == '432425197609135822' and cpy_PID == 'AB3DB3EDC74B4789BFECEC5743B39929') or \
            (orgPID == '432925194711116920' and cpy_PID == '8E779D0A38FC41AEBF2C2FBE6E5815AF') or \
            (orgPID == '440520196701272810' and cpy_PID == 'B00765F89E324C98B8884CA1AF287FD9') or \
            (orgPID == '440582198803020639' and cpy_PID == '5D923ED1698F418DA36FF56578554CFE') or \
            (orgPID == '440583199001024516' and cpy_PID == '62FFBBE15BD447C18293BDF756D9D25F') or \
            (orgPID == '440881198002257214' and cpy_PID == '26E533EF76834B6F8275008F8FA4EFEA') or \
            (orgPID == '440882198202211110' and cpy_PID == '30C7C949C4F140DAA196ED2536E97E48') or \
            (orgPID == '440982199402011471' and cpy_PID == 'A46134790C964A9786C67406742AB4F0') or \
            (orgPID == '450126199509016122' and cpy_PID == '3D83EC7F57C844C38736746B60BD59B7') or \
            (orgPID == '450521198111255217' and cpy_PID == 'F688B9E3C0554A399A218AF849D8E1A4') or \
            (orgPID == '452124196806182771' and cpy_PID == 'B1C7746596544B7C9D5AB4ED9916CFDD') or \
            (orgPID == '452226199606200017' and cpy_PID == 'B92E0127F501495AB7F08D8E5A10AF12') or \
            (orgPID == '452623197907062416' and cpy_PID == '156984236C524460B3C57463AF56E8D3') or \
            (orgPID == '460035198210152515' and cpy_PID == '4F77359C5A6D4BEDA6E1443721BA3C15') or \
            (orgPID == '513721198908253390' and cpy_PID == '4EE2CDD0B7394736AE2156CC6DCA70B4') or \
            (orgPID == '310102196209091272' and cpy_PID == 'FC6A3518F97B4444BA6E30C64806CFD0') or \
            (orgPID == '310110196109044217' and cpy_PID == 'A0363892CA9E47E98666A3BB5CE6D060') or \
            (orgPID == '310224197507239317' and cpy_PID == 'ADF0B987FD8F4ED5A77CD176BBA412F5') or \
            (orgPID == '432925194711116920' and cpy_PID == '2D98E23A0224460FAED2DA0B31F5FA5F') or \
            (orgPID == '310110196012243244' and cpy_PID == '719E6ACFFEA241C8B470E404679B9E8A') or \
 \
            (orgPID == '132627196312308612' and cpy_PID == '6B27B584F64B45D3B95B9ACF5842A7BF') or \
            (orgPID == '310102196209091272' and cpy_PID == 'C0E4DE979BEB4EA9A5ACD7BB12C13C3C') or \
            (orgPID == '310103194501102415' and cpy_PID == '2ABB86E9594F4054873EEF75283B5620') or \
            (orgPID == '31011019530117245X' and cpy_PID == '074F7E8AEECA44629B1563253AE837B3') or \
            (orgPID == '310221195201203612' and cpy_PID == 'C229275B5EE24786B3EF12B837C48C9F') or \
            (orgPID == '310222193410113610' and cpy_PID == '8486F4004F004C6C8B03B6773D52F793') or \
            (orgPID == '310222193410113610' and cpy_PID == '8486F4004F004C6C8B03B6773D52F793') or \
            (orgPID == '310222193410113610' and cpy_PID == '8486F4004F004C6C8B03B6773D52F793') or \
            (orgPID == '31022219640615363X' and cpy_PID == '48DAD73262634E5E845C1237D3F46193') or \
            (orgPID == '320802199001221510' and cpy_PID == 'EDB853B04345463CA0A6D17CF37F6412') or \
            (orgPID == '330211197004250719' and cpy_PID == '85A2048EA163464183E2165AFA853335') or \
            (orgPID == '330211197004250719' and cpy_PID == '85A2048EA163464183E2165AFA853335') or \
            (orgPID == '340221197612122870' and cpy_PID == 'CA9FF646A0C44B14B64E3E75462F12E3') or \
            (orgPID == '360121199403082443' and cpy_PID == '371824D81A62459D9DF3D73F89B14404') or \
            (orgPID == '370402198104050649' and cpy_PID == '0F5D33CA436E4B5794EABBD63D6AA537') or \
            (orgPID == '379014197404216054' and cpy_PID == '7BB62098E5F54871AFBA6AAAF1261813') or \
            (orgPID == '432925194711116920' and cpy_PID == '0953B53CE4424F43AB94F9E7FE81FFF8') or \
            (orgPID == '510322198010033836' and cpy_PID == '6C58703A09F44D2D84A4890963D5F746') or \
            (orgPID == '51162119870920429X' and cpy_PID == '1E4C2743204F43CBA5AC9B5547DA23A0') or \
            (orgPID == '51162119870920429X' and cpy_PID == '1E4C2743204F43CBA5AC9B5547DA23A0') or \
            (orgPID == '513231198012220313' and cpy_PID == 'A951EBEF153C48A2BEB910294B3DB058') or \
            (orgPID == '310228198104046019' and cpy_PID == 'C99F212EA65B4A22BE6AEFC728DA69F1') \
            :
        return True
    else:
        return False


def special_handling2(orgPID,orgIMGID,cpy_PID): #处理错误标
    #return False
    if (orgPID=='310108197801294434' and orgIMGID=='17' and cpy_PID=='310107197203201211')\
            or (orgPID == '310113197403192417' and orgIMGID == '5' and cpy_PID == '310111197103170416')\
            or (orgPID == '310114198209140032' and orgIMGID == '5' and cpy_PID == '310222196710010211')\
            or (orgPID == '310222197312170214' and orgIMGID == '7' and cpy_PID == '310114197704033213'):
        return True
    else:
        return False
def special_handling3(orgPID,cpy_PID):
    if (orgPID == '372923198506065975'and  cpy_PID== '372923198708143812') or \
    (orgPID == '320826198910202217' and cpy_PID == '31010219890602121X') or \
    (orgPID == '452130198601203010' and cpy_PID == '310115199307122912') or \
    (orgPID == '320625197909216695' and cpy_PID == '310110198010293255') or \
    (orgPID == '310110197412140812' and cpy_PID == '310109198312283591') :
        return True
    elif( orgPID == '372923198506065975'and cpy_PID== '372923198506065975') or \
        (orgPID == '320826198910202217' and cpy_PID == '320826198910202217') or \
        (orgPID == '452130198601203010' and cpy_PID == '452130198601203010') or \
        (orgPID == '320625197909216695' and cpy_PID == '320625197909216695') or \
        (orgPID == '310110197412140812' and cpy_PID == '310110197412140812'):
        return False
    elif orgPID == cpy_PID:
        return True
    return False

def kcf_test(threshold = 0.927,resaveimgflag=False):
    all=0
    right=0
    wrong=0

    for line in retlist:
        line = line.strip()
        orgPID, orgImg , kcf_PID , kcf_IMG ,kcf_dist = line.split(',')

        all=all+1

        if resaveimgflag and kcf_PID=='' and kcf_IMG=='':  #未召回
            reSaveImg(orgPID, orgImg, kcf_PID, kcf_IMG, 'kc', threshold, 'unrecall')

        #kcf
        if special_handling3(orgPID,kcf_PID) or special_handling(orgPID,kcf_PID)  or special_handling2(orgPID,orgImg,kcf_PID):
            if float(kcf_dist) < threshold:
                #正确召回
                right=right+1

            continue

        if orgPID != kcf_PID and kcf_PID!='':
            if float(kcf_dist) < threshold:
                #误识召回
                wrong=wrong+1
                print(orgPID, kcf_PID)



                if resaveimgflag :
                    reSaveImg(orgPID, orgImg, kcf_PID, kcf_IMG, 'kc', threshold, 'error')

    print('暴力库_threshold:',threshold,'召回率：',right/all,'误识率:',wrong/all)


def st_test(threshold=0.927,resaveimgflag=False):
    all = 0
    right = 0
    wrong = 0

    for line in retlist:
        line = line.strip()
        orgPID, orgIMG, \
        _, kcf_similarity, kcf_dist, kcf_PID, kcf_IMG, \
        _, st_similarity, _, st_PID, st_IMGID, \
        _, yt_similarity, _, yt_PID, yt_IMGID = line.split(',')

        all = all + 1

        if resaveimgflag and st_PID=='' and st_IMGID=='':  #未召回
            reSaveImg(orgPID, orgIMG, st_PID, st_IMGID.strip(), 'st', threshold, 'unrecall')

        # st
        if orgPID == st_PID or special_handling(orgPID,st_PID)  or special_handling2(orgPID,orgIMG,st_PID) :
            if float(st_similarity) > threshold:
                # 正确召回
                right = right + 1

            continue

        if orgPID != st_PID and st_PID!='':
            #print(st_similarity)
            if float(st_similarity) > threshold:
                # 误识召回
                wrong = wrong + 1
                if resaveimgflag :
                    reSaveImg(orgPID, orgIMG, st_PID, st_IMGID.strip(), 'st', threshold, 'error')

    print('商汤_threshold:', threshold, '召回率：', right / all, '误识率:', wrong / all)

def yt_test(threshold=0.927,resaveimgflag=False):
    all = 0
    right = 0
    wrong = 0

    for line in retlist:
        line = line.strip()
        orgPID, orgIMG, \
        _, kcf_similarity, kcf_dist, kcf_PID, kcf_IMG, \
        _, st_similarity, _, st_PID, st_IMGID, \
        _, yt_similarity, _, yt_PID, yt_IMGID = line.split(',')

        all = all + 1

        if resaveimgflag and yt_PID=='' and yt_IMGID=='':  #未召回
            reSaveImg(orgPID, orgIMG, yt_PID, yt_IMGID.strip(), 'yt', threshold, 'unrecall')

        # yt
        if orgPID == yt_PID or special_handling(orgPID,yt_PID) or special_handling2(orgPID,orgIMG,yt_PID):
            if float(yt_similarity) > threshold:
                # 正确召回
                right = right + 1

            continue

        if orgPID != yt_PID and yt_PID!='':
            #print(yt_similarity)
            if float(yt_similarity) > threshold:
                # 误识召回
                wrong = wrong + 1
                if resaveimgflag :
                    reSaveImg(orgPID, orgIMG, yt_PID, yt_IMGID.strip(), 'yt', threshold, 'error')

    print('依图_threshold:', threshold, '召回率：', right / all, '误识率:', wrong / all)


kcf_test(0.926)
# st_test(0.9,True)
# yt_test(0.9,True)

for iii in range(700, 1000,1):
    if iii<860 or iii>960:
        if iii % 20 !=0:
            continue
    threshold = iii / 1000
    kcf_test(threshold)

print('\n')
# for iii in range(700, 1000,1):
#     # if iii<880 or iii>980:
#     if iii % 10 !=0:
#         continue
#     threshold = iii / 1000
#     st_test(threshold)
#
# print('\n')
# for iii in range(700, 1000,1):
#     # if iii<880 or iii>980:
#     if iii % 10 !=0:
#         continue
#     threshold = iii / 1000
#     yt_test(threshold)