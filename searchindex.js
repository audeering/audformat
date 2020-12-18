Search.setIndex({docnames:["accessing-data","api","api-define","api-errors","api-testing","api-utils","changelog","combine-tables","contributing","create-database","data-conventions","data-example","data-format","data-header","data-introduction","data-tables","emodb-example","genindex","index","install","map-scheme"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["accessing-data.rst","api.rst","api-define.rst","api-errors.rst","api-testing.rst","api-utils.rst","changelog.rst","combine-tables.rst","contributing.rst","create-database.rst","data-conventions.rst","data-example.rst","data-format.rst","data-header.rst","data-introduction.rst","data-tables.rst","emodb-example.rst","genindex.rst","index.rst","install.rst","map-scheme.rst"],objects:{"":{audformat:[1,0,0,"-"]},"audformat.Column":{get:[1,2,1,""],rater_id:[1,3,1,""],scheme_id:[1,3,1,""],set:[1,2,1,""]},"audformat.Database":{__contains__:[1,2,1,""],__getitem__:[1,2,1,""],__setitem__:[1,2,1,""],drop_files:[1,2,1,""],drop_tables:[1,2,1,""],expires:[1,3,1,""],files:[1,2,1,""],languages:[1,3,1,""],load:[1,2,1,""],load_header_from_yaml:[1,2,1,""],map_files:[1,2,1,""],media:[1,3,1,""],name:[1,3,1,""],pick_files:[1,2,1,""],pick_tables:[1,2,1,""],raters:[1,3,1,""],save:[1,2,1,""],schemes:[1,3,1,""],segments:[1,2,1,""],source:[1,3,1,""],splits:[1,3,1,""],tables:[1,3,1,""],usage:[1,3,1,""]},"audformat.Media":{bit_depth:[1,3,1,""],channels:[1,3,1,""],format:[1,3,1,""],sampling_rate:[1,3,1,""],type:[1,3,1,""],video_channels:[1,3,1,""],video_depth:[1,3,1,""],video_fps:[1,3,1,""],video_resolution:[1,3,1,""]},"audformat.Rater":{type:[1,3,1,""]},"audformat.Scheme":{draw:[1,2,1,""],dtype:[1,3,1,""],is_numeric:[1,2,1,""],labels:[1,3,1,""],maximum:[1,3,1,""],minimum:[1,3,1,""],to_pandas_dtype:[1,2,1,""]},"audformat.Split":{type:[1,3,1,""]},"audformat.Table":{columns:[1,3,1,""],copy:[1,2,1,""],df:[1,2,1,""],drop_columns:[1,2,1,""],drop_files:[1,2,1,""],drop_index:[1,2,1,""],ends:[1,2,1,""],extend_index:[1,2,1,""],files:[1,2,1,""],get:[1,2,1,""],index:[1,2,1,""],is_filewise:[1,2,1,""],is_segmented:[1,2,1,""],load:[1,2,1,""],media_id:[1,3,1,""],pick_columns:[1,2,1,""],pick_files:[1,2,1,""],pick_index:[1,2,1,""],save:[1,2,1,""],set:[1,2,1,""],split_id:[1,3,1,""],starts:[1,2,1,""],type:[1,3,1,""]},"audformat.define":{DataType:[2,1,1,""],Gender:[2,1,1,""],IndexField:[2,1,1,""],IndexType:[2,1,1,""],MediaType:[2,1,1,""],RaterType:[2,1,1,""],SplitType:[2,1,1,""],TableStorageFormat:[2,1,1,""],Usage:[2,1,1,""]},"audformat.define.DataType":{BOOL:[2,3,1,""],DATE:[2,3,1,""],FLOAT:[2,3,1,""],INTEGER:[2,3,1,""],STRING:[2,3,1,""],TIME:[2,3,1,""]},"audformat.define.Gender":{CHILD:[2,3,1,""],FEMALE:[2,3,1,""],MALE:[2,3,1,""],OTHER:[2,3,1,""]},"audformat.define.IndexField":{END:[2,3,1,""],FILE:[2,3,1,""],START:[2,3,1,""]},"audformat.define.IndexType":{FILEWISE:[2,3,1,""],SEGMENTED:[2,3,1,""]},"audformat.define.MediaType":{AUDIO:[2,3,1,""],OTHER:[2,3,1,""],VIDEO:[2,3,1,""]},"audformat.define.RaterType":{HUMAN:[2,3,1,""],MACHINE:[2,3,1,""],OTHER:[2,3,1,""],TRUTH:[2,3,1,""],VOTE:[2,3,1,""]},"audformat.define.SplitType":{DEVELOP:[2,3,1,""],OTHER:[2,3,1,""],TEST:[2,3,1,""],TRAIN:[2,3,1,""]},"audformat.define.TableStorageFormat":{CSV:[2,3,1,""],PICKLE:[2,3,1,""]},"audformat.define.Usage":{COMMERCIAL:[2,3,1,""],OTHER:[2,3,1,""],RESEARCH:[2,3,1,""],RESTRICTED:[2,3,1,""],UNRESTRICTED:[2,3,1,""]},"audformat.errors":{BadIdError:[3,1,1,""],BadTypeError:[3,1,1,""],BadValueError:[3,1,1,""]},"audformat.testing":{add_table:[4,4,1,""],create_audio_files:[4,4,1,""],create_db:[4,4,1,""]},"audformat.utils":{concat:[5,4,1,""],map_language:[5,4,1,""],read_csv:[5,4,1,""],to_filewise_index:[5,4,1,""],to_segmented_index:[5,4,1,""]},audformat:{Column:[1,1,1,""],Database:[1,1,1,""],Media:[1,1,1,""],Rater:[1,1,1,""],Scheme:[1,1,1,""],Split:[1,1,1,""],Table:[1,1,1,""],define:[2,0,0,"-"],errors:[3,0,0,"-"],filewise_index:[1,4,1,""],index_type:[1,4,1,""],segmented_index:[1,4,1,""],testing:[4,0,0,"-"],utils:[5,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"000728180":11,"001":[0,4,7,11,20],"002":[0,4,7,11,20],"003":[7,11,20],"003180717":11,"004":[7,11,20],"004521773":11,"005":[11,20],"005465989":11,"006":11,"006250":16,"007":11,"008":11,"009":11,"010":11,"010000":11,"011":11,"012":11,"013":11,"013926055":11,"014":11,"014356578":7,"015":11,"016":11,"017":11,"017031201":11,"018":11,"019":11,"020":11,"021":11,"021853186":11,"022":11,"023":11,"024":11,"025":11,"026":11,"027":11,"028":11,"029":11,"030":11,"030000":11,"031":11,"032":11,"032772107":7,"033":11,"034":11,"035":11,"036":11,"036863225":11,"037":11,"038":11,"039":11,"039833436":11,"03a01fa":16,"03a01nc":16,"03a01wa":16,"03a02fc":16,"03a02nc":16,"03a02ta":16,"03a02wb":16,"03a02wc":16,"03a04ad":16,"03a04fd":16,"040":11,"040000":11,"041":11,"042":11,"043":11,"044":11,"045":11,"045900226":11,"046":11,"047":11,"047302681":11,"048":11,"049":11,"049461405":11,"050":11,"051":11,"052":11,"053":11,"054":11,"055":11,"056":11,"057":11,"058":11,"059":11,"05msekapg5":11,"060":11,"061":11,"061663934":11,"061980123":11,"062":11,"063":11,"064":11,"065":11,"065215625":11,"066":11,"067":11,"067828060":7,"068":11,"068558322":11,"069":11,"070":11,"070000":11,"071":11,"071303407":11,"072":11,"073":11,"074":11,"075":11,"076":11,"077":11,"078":11,"079":11,"080":11,"080000":11,"081":11,"082":11,"083":11,"084":11,"085":11,"085254200":11,"085448723":11,"086":11,"087":11,"088":11,"089":11,"090":11,"090000":11,"091":11,"092":11,"093":11,"094":11,"095":11,"096":11,"097":11,"097981251":7,"098":11,"099":11,"0q0bio1vhc":11,"100":[11,16],"1000":10,"100000":11,"101233972":7,"101653192":7,"110":11,"116079916":11,"11b03lc":16,"120":11,"120000":11,"123625":16,"127375815":11,"129017352":11,"12aec1be13":11,"130":[11,16],"130000":11,"134665293":11,"136544369":7,"137665019":11,"140000":11,"147384615":11,"147777812":11,"147834827":11,"147981755":11,"150":11,"150000":11,"150127366":11,"151015934":11,"151598082":11,"156176763":11,"15b01na":16,"15b09ac":16,"16000":[1,4,11,16],"160000":11,"161033122":11,"170":11,"170000":11,"170777126":11,"175673625":11,"177922724":11,"179648288":11,"180":11,"180000":11,"181927969":7,"190":11,"190000":11,"190835736":7,"193273377":11,"194464484":11,"194513392":11,"1970":11,"1997":16,"1999":16,"1jr2cwfcej":11,"1st":13,"1xxaj1zfyz":11,"1ys7ddeoxt":11,"200000":11,"204194646":11,"206291045":11,"210":11,"212937779":11,"213476224":7,"214421088":11,"214509967":11,"216848840":11,"217188112":11,"220":11,"220000":11,"223235973":11,"228089789":11,"228883349":11,"229159403":11,"230":11,"235332300":7,"240":11,"240000":11,"244027967":11,"244067209":11,"246061887":11,"246409405":11,"246545116":11,"247088422":11,"250000":11,"251397263":11,"260":11,"260000":11,"267883418":11,"270000":11,"272322253":11,"272665806":11,"279174704":11,"280":11,"280000":11,"282444637":11,"284250498":11,"290":11,"290000":11,"297519767":7,"298234003":11,"2eqsx3ir1d":11,"2irb9ef1ka":11,"2kvtayrxui":11,"2n5vm9fjmj":11,"2rjxfckkfg":11,"2tv1hmhwj5":11,"2w4eboesiv":11,"300":11,"300000":11,"305450663":11,"310000":11,"311299187":11,"320000":11,"322573359":11,"329576261":11,"330000":11,"331616260":11,"335742193":11,"340":11,"340000":11,"344311784":11,"347574714":11,"349810085":7,"349915447":11,"350":11,"350000":11,"357450013":11,"35qsizz2qt":11,"360000":11,"362499320":11,"366089340":7,"370":11,"370000":11,"372157802":7,"377531827":11,"378488800":11,"380":11,"380000":11,"380182833":11,"386050003":11,"386572769":11,"386829440":11,"388139246":11,"388580851":11,"390":11,"390000":11,"390302083":11,"394325906":11,"399836036":7,"3bimnwjtfo":11,"400":11,"400000":11,"408427802":11,"40ftqu35fu":11,"40pv7kfd25":11,"410":11,"410000":11,"411900972":11,"420":11,"420000":11,"421277120":7,"422345346":11,"422875624":11,"425714298":11,"426472318":11,"430":11,"430000":11,"431207022":11,"435575869":11,"439812500":16,"440":11,"440000":11,"447567197":11,"450":11,"450000":11,"450280201":7,"450543924":11,"460000":11,"466720697":7,"470000":11,"476627794":7,"480":11,"480000":11,"481957805":7,"486389681":11,"490":11,"490000":11,"496560309":11,"498062500":16,"4bizvhngec":11,"4bo3trc9a2":11,"4h7esk7nfr":11,"4l5m8fmhxx":11,"4tqbd8tjiv":11,"500":[10,11,16],"500000":10,"502764061":7,"504874999":16,"510":11,"510000":11,"512621427":11,"516025248":11,"517619067":11,"520000":11,"520589333":11,"522645241":7,"522947785":7,"523502679":7,"527303234":11,"530":11,"530000":11,"534126505":11,"537847272":11,"538153223":7,"540":11,"540000":11,"543692849":11,"546983628":11,"548741989":11,"549639297":11,"550000":11,"552256338":7,"559092162":11,"560":11,"560000":11,"562418673":11,"564185525":11,"564250999":7,"566795542":11,"570":11,"572360527":11,"573054910":11,"577123577":11,"580031562":11,"580591061":11,"584147673":11,"590":11,"590000":11,"590438238":11,"590984677":11,"591434600":11,"594179859":11,"595598534":11,"5agrur6oaq":11,"5ayi4z7grl":11,"5ivn01iu6v":11,"5j9kgwwmmn":11,"5p6gkeomap":11,"5xe4srvvao":11,"5ydyhp8gck":11,"600":[1,11],"600000":11,"603056008":11,"607925127":11,"608702420":7,"608767081":11,"60s":4,"610":11,"611250":16,"612027002":11,"612398498":11,"615252126":11,"616808213":11,"618380694":11,"627225009":11,"630":11,"630000":11,"639":5,"640":11,"640000":11,"647921331":11,"648927644":11,"648946607":11,"649523138":11,"650":11,"650000":11,"655594954":11,"656811605":11,"659787252":11,"660":11,"660000":11,"669643549":11,"66dhpnad0r":11,"670":11,"670000":11,"676755470":11,"679255686":11,"680":11,"680000":11,"681055415":7,"681803747":11,"690":11,"696812500":16,"6bevyittab":11,"6gm35q687k":11,"6j1mxilzd1":11,"6ksblgfain":11,"6mjn9g4f1t":11,"6yu2sxufrf":11,"700":11,"700000":11,"702513806":11,"702677042":7,"705169463":7,"714826306":11,"715839928":7,"720":11,"720000":11,"722908334":11,"723102824":11,"727345778":7,"730":11,"735687500":16,"737320267":11,"738224611":11,"740":11,"740000":11,"742688450":11,"744689119":11,"749593501":11,"749860981":7,"750":11,"750000":11,"751600565":11,"756127186":11,"756417121":11,"760":11,"760000":11,"760964574":11,"762413518":11,"770":11,"770000":11,"770251090":11,"770323780":11,"770880434":11,"774424694":11,"780000":11,"780629261":11,"781680493":11,"784946888":11,"786161116":11,"786251617":7,"786naikykw":11,"790":11,"790000":11,"795988223":11,"7e1fdx0lcm":11,"7jfs61ej1":11,"7nuw5bbfmw":11,"7ozpnygmss":11,"7s92i5nixb":11,"800":[1,11],"800000":11,"800021605":11,"802517899":11,"810":11,"810476728":11,"816604945":11,"817832628":11,"820":11,"820000":11,"827209927":7,"828050921":11,"830":11,"830000":11,"831157584":11,"831808057":11,"839681291":11,"843861113":11,"847199235":11,"850":11,"850000":11,"8506":16,"860":11,"860000":11,"870":11,"876714854":11,"877812500":16,"880":11,"880000":11,"885017371":11,"886075829":7,"890":11,"890000":11,"890340242":11,"894564314":11,"897191672":11,"898250":16,"899934873":11,"8dmr3mxke5":11,"8e3hcla5o6":11,"8hwuejpcoh":11,"8im0u3dsl9":11,"900":11,"900000":11,"908667721":11,"910":11,"910000":11,"912448401":11,"919952138":11,"920":11,"920000":11,"920304336":11,"92ocyo9jjh":11,"930":11,"930000":11,"931852540":11,"932367395":11,"935262547":11,"935963220":11,"936774836":7,"940":11,"940000":11,"942166010":11,"947547474":11,"950":11,"950000":11,"959599940":11,"960":11,"960000":11,"970000":11,"972972833":11,"975147868":7,"976289027":11,"980":11,"982484053":11,"983153869":11,"986365615":11,"990":11,"990000":11,"9bmwbzzu5h":11,"9gdem47faj":11,"9igf6fewut":11,"9jvffroxap":11,"9rvwy3bsok":11,"case":[1,10,14],"class":[1,2,3],"default":[1,4,13],"f\u00fcr":16,"float":[1,2,4,7,9,10,11,13,16],"function":[1,4,16],"holzst\u00fcck":16,"import":[0,4,5,7,9,10,11,13,15,16,20],"int":[0,1,2,4,5,9,11,16],"k\u00f6nnte":16,"long":10,"new":[1,10],"return":[0,1,4,5,20],"short":[0,13],"st\u00fcck":16,"static":1,"t\u00fcten":16,"true":[1,7,9,15,16,20],"try":1,Added:6,Age:16,And:20,But:20,Das:16,For:[0,1,4,9,16],IDS:1,IDs:[1,20],The:[4,5,6,7,8,9,10,12,13,14,16,20],Their:16,Then:16,There:[1,15],These:13,Use:10,Was:16,__contains__:1,__getitem__:1,__setitem__:1,a01:16,a02:16,a04:16,a05:16,a07:16,a62b4vf0o8:11,a7sdsgj20g:11,aampmkfbqj:11,abend:16,abgeben:16,about:[12,13,16,20],absolut:[0,5],abspath:0,access:[10,12,13,14,15,16],accur:10,acoust:16,across:[12,14],act:16,action:5,activ:[8,19],actor:16,actual:[12,20],add:[0,1,4,9,13,15,16],add_tabl:[7,20],adding:10,addit:[1,9,10,13,16,20],adher:6,aeltxtxir0:11,after:10,afterward:16,age:[1,7,10,14,16,20],agent:10,aggreg:[10,14],agn:16,agreement:16,ahbhsuisbp:11,all:[0,1,6,7,8,9,10,14,15,16],allow:[1,4,13,14],also:[1,5,8,10,20],alwai:[1,8,10],aml1gonwk1:11,amplitud:4,anecho:16,anger:[9,16],ani:[1,3,20],annot:[0,1,8,9,12,14,18],anoth:10,appear:13,append:10,appli:[1,4,5,13,14],applic:14,aqmnyzukw6:11,ar73ede25r:11,arbitrari:[13,15],arg:5,argument:[1,5,10,16,20],around:9,arous:10,ask:16,asr:10,assign:[1,13,15,16],associ:10,assum:[9,10],astyp:16,attribut:20,audeer:[1,8,9,10,11,13,14,16],audformat:[0,7,8,9,10,11,14,19,20],audio:[0,1,2,4,5,7,11,12,13,14,16,18,20],audiofil:[10,16],auf:16,author:1,automat:[5,8,10],avail:[1,7,8,16],averag:[10,11],avi:11,b01:16,b02:16,b03:16,b09:16,b0s8lle3hc:11,b10:16,b263travlv:11,badiderror:1,badvalueerror:1,bar:5,base:[1,6],basename_wo_ext:16,basic:16,bbye17hvbp:11,been:15,befindet:16,befor:16,begin:15,belong:[7,10,13,16],below:13,benlglzs0b:11,berlin:16,besucht:16,between:12,biecewi5aw:11,bilderbar:16,bin:[8,16,19],bit:[1,13],bit_depth:[1,11,13],bjbzskafbn:11,blank:10,bool:[1,2,4,5],boredom:16,both:7,bound:10,bqvulwahn7:11,bsgdbao80w:11,by2sftpnkr:11,calcul:10,call:4,callabl:[1,4],can:[0,1,4,7,8,9,10,12,13,14,15,16,18,20],categor:14,categori:10,categoricaldtyp:1,caus:3,caxipvpdmk:11,cfxeiotxla:11,challeng:14,chamber:16,chang:[1,6,8],changelog:8,channel:[1,4,11,13,16],charact:10,character:13,characterist:15,check:[1,8,16],child:2,chjxfgbrk3:11,chuj8achl7:11,citeseerx:16,cj18dwaju7:11,cj6qf98pql:11,cknuzdmweg:11,classifi:[1,11],client:10,clone:8,code:[0,9,16],collect:[10,16],column:[0,2,4,7,9,10,11,12,15,16,20],column_id:1,columnid:13,com:[1,8,9,11,13],combin:1,combined_t:7,come:15,command:9,commerci:[1,2,13],commit:8,common:10,compare_2016:11,concaten:5,condit:1,confid:[1,14,16],conform:[1,5],connect:12,consid:[1,10],consist:[1,10,15],contain:[1,5,9,10,12,16,18],content:[2,10,12],continu:1,convers:5,convert:[0,1,5,12,16],copi:1,copytre:16,core:13,corpu:14,correct:16,correspond:18,could:16,cover:9,crdwbiuumh:11,creat:[1,4,5,10,13,14,15,19],create_db:[0,7,9,11,15,20],crucial:1,cseg:10,csv:[1,2,5,11,12,16],d3vgz9ciai:11,d5cxxyjock:11,dai:[5,7,10,11,15,16],dann:16,das:16,data:[1,2,5,9,13,14,18,20],databas:[2,4,7,11,14,15,18,20],databasenam:13,datafram:[0,1,5,7,11,12,15],datatyp:[1,7,9,10,13,16],date:[1,2,8,9,11,14],datetim:10,datetime64:1,db_dir:16,db_minim:9,dbc4uxaxez:11,decim:16,def:16,defin:[1,3,4,7,9,10,13,16,18,20],definit:[1,2,12],delim_whitespac:16,dem:16,den:16,denn:16,depart:16,depth:[1,13],der:16,deriv:1,describ:16,descript:[1,11,13,15,16],detect:5,determin:4,deu:[1,9,11,16],dev:[2,9,10,11],develop:[1,2,10],df_age:7,df_likabl:7,dfg:16,dfuvjqz2d:11,dict:[1,3,4],dictionari:[1,3,4,13,20],die:16,differ:[1,4,7,10,13,16],dimens:10,directli:[0,1],directori:[1,4,5,8],disgust:16,disk:[1,9,18],displai:7,dlalsrlibm:11,dmv50hgstj:11,doc:8,document:[6,10,16],doe:[1,10],doi:16,domin:10,don:7,download:16,dpcpttjbnu:11,draw:[1,4],drop:1,drop_column:1,drop_fil:1,drop_index:1,drop_tabl:1,drpgmlj2t:11,dtype:[0,1,5,9,10,11,13,16],dummi:[9,10],durat:[4,16],dvw3twf3cx:11,dyh8lohdau:11,e64legvbti:11,each:[5,9,10,12,13,15,16],easili:14,eben:16,edu:16,eisschrank:16,either:[1,15],ejgqhcb5gx:11,element:13,elif:1,els:16,emot:[1,9,10,12,16],emotion_map:16,empti:[1,4,15],encod:[16,20],end:[1,2,5,7,11,15],eng:[1,5,9,11],english:[1,5],enough:[10,14],entri:[1,9,12,16],env:[8,19],environ:[8,19],erkannt:16,erkennung:16,erklaerung:16,error:[5,8],esgujlrdrv:11,eske1n8e3a:11,etc:[4,14],evalu:16,even:[5,8],everi:[1,4,10,13,16],everyon:8,ew2gkscjaa:11,exampl:[0,1,5,7,9,10,12,20],execut:8,exist:[1,16],expect:[1,3],expected_typ:3,experi:14,expir:[1,13],expiri:1,explicitli:[4,13],express:16,extend:[1,10,16],extend_index:1,extens:1,extract_arch:16,ezlnrz97cc:11,f31kxejwc7:11,f9m9pygqtc:11,fals:[1,4,5,10,16],fdpqz8jspl:11,fear:16,featur:11,feel:8,felt:16,femal:[2,10,16,20],few:8,fhcierronk:11,field:[1,2,3,13,20],file:[1,2,4,5,6,7,8,11,12,13,14,15,16,18,20],file_dur:4,file_root:4,filewis:[1,2,4,5,7,9,10,11,13,16,20],filewise_index:[5,10,13,15,16],fill:[1,9],fill_valu:1,filter:14,find:8,first:[4,16],fit:[1,13],fkxuakiq1k:11,flac:1,float64:5,fly:10,folder:[1,4,5,9,12,16,18],follow:[0,1,7,8,9,10,12,14,15,16,20],foo:9,format:[1,2,4,6,9,11,13,14,16,18],fpdauxdnzw:11,fpn56sa45z:11,frame:[1,5,13],free:8,from:[1,4,5,8,9,15,16],from_i:16,func:1,fund:16,further:[14,16],fy46hvpyzz:11,g5vxvyrwzw:11,gdvtbbr8tj:11,gefahren:16,gehen:16,gender:[1,10,14,16,20],gener:[4,8,14],gerad:16,german:16,get:[0,1,4,7,8,10,11,15,20],getragen:16,git:8,github:[8,9,11],gitlab:13,gold:[9,11,16],gold_id:10,gold_standard:10,goodby:20,gqc8cnf2rg:11,ground:2,gtmgq4xgyl:11,gxi4hhl1yl:11,gyny10bj92:11,h0wryaqq5j:11,h1w5jwuzuh:11,h5urpwez6w:11,h8geva9brt:11,habe:16,haben:16,had:16,handl:14,happen:7,happi:16,has:[1,3,5,9,10,15],haus:16,have:[0,1,4,7,9,10,13,16,18,20],header:[1,10,11,12,18],header_onli:1,height:13,hello:20,here:[10,20],heut:16,hg4vqlbqli:11,highlight:[7,12],hinlegen:16,hnk4wc09ka:11,ho0w8oo5v3:11,ho22mqtiit:11,hoch:16,hochgetragen:16,hold:[1,10,12,20],home:[8,19],host:13,how:[0,9],html:8,http:[1,8,9,11,13,16],human:[1,2,9,10,11,13,14,16],hyboxown5k:11,i4esad9pij:11,ibveuu5hq9:11,ic552w0j6g:11,ich:16,identifi:[1,3,13],ihm:16,iloc:0,im05vpipy6:11,immer:16,implement:[12,18],improv:8,imqlgp8vnt:11,includ:[1,20],inconsist:8,increas:10,indent:1,index:[1,2,5,7,9,10,12,15,16],index_col:16,index_ex:1,index_typ:[4,7],indexfield:1,indextyp:[1,4,7,20],individu:14,info:16,inform:[0,1,7,9,12,13,14,16,20],initi:6,inplac:1,input:[1,4,5],insid:14,instal:18,instanc:12,instead:[8,10],instruct:18,int64:1,integ:[1,2,4,7,9],intend:18,interest:[4,20],intern:[9,11],invalid:[1,4],invalid_id:3,invalid_valu:3,invit:8,ipiohkdtih:11,iqyg59h0kr:11,ir7g3drgh:11,irjod5ljga:11,is_filewis:1,is_numb:16,is_numer:1,is_seg:1,iso:5,issu:8,ist:16,iter:1,itlcyo3k3:11,its:[1,16],iuqmygsazq:11,iwvpilvluo:11,ixhta9eg73:11,izrqtt5h6c:11,j6pqeuu8ct:11,jadjmntal9:11,jbdfvjp78e:11,je2u1wsmjl:11,jetzt:16,jjzfiitcxc:11,jmddawnsv:11,jmdwmmveq1:11,jn01fm1skq:11,join:16,jrtxw94ibm:11,just:[7,9],jwrlej2hzi:11,k22moxjwrn:11,k9tqaluaez:11,karl:16,kclfwcu5vi:11,kcmbbyzxyq:11,kdeshgfvs4:0,kdfi7fzdl:11,keep:[1,6,20],kei:[1,4,13,16,20],keyword:5,khxwl7vmql:11,kijxlwzjhu:11,kky9fqiemo:11,kpb54szxd7:11,ku4pgrkt9c:11,kwarg:5,kzgnlze8sz:11,l1x2d5z7qz:11,l6zgr3m3dl:11,la224zmuk0:11,label1:11,label2:11,label3:11,label:[0,1,7,9,10,11,13,14,15,16],label_map_int:[9,11],label_map_str:[9,11],lablaut:16,labsilb:16,lambda:[10,16],languag:[1,5,9,10,11,13,16],lappen:16,later:10,latest:8,latin:16,learn:14,least:5,len:1,length:1,let:9,level:[1,10],lhyvs5vq2:11,librispeech:10,libsvm:11,liegt:16,likabl:7,like:[10,16],likeabl:7,likewis:20,limit:1,link:[1,8,12,13,14,15],linkcheck:8,list:[1,3,4,10,12,13,16,20],listdir:16,lmnjq419cx:11,load:[1,16,18],load_data:1,load_header_from_yaml:1,loc:16,locat:[1,14,16],look:[0,9,16,18],lower:[1,10],lowercas:10,lxho9vmofu:11,lxr6zva0k4:11,lxr76nmjmr:11,lyisth2vwz:11,lyvwukzzhh:11,m87wq1hvqy:11,m9alyi0s5x:11,ma1kbvnqkd:11,machin:[2,11,14],made:[8,11],mai:14,main:14,make:[8,14,16],male:[2,10,16,20],mandatori:13,mani:14,map:[1,5,10,16],map_fil:[0,1],map_languag:16,mark:13,match:1,maximum:[1,7,9,10,11,13,16],mazuztdcvi:11,mean:[10,15],media:[2,4,9,11,12,15,16,18],media_id:[1,4,9,11,13],mediaid:13,mediatyp:[1,9],merg:14,meta:[1,10,12,13,14,16],metadata:[10,14],method:0,metric:10,mfa:10,mfjki1mvoq:11,mgjnphxgjm:11,microphon:[9,11,16],might:10,miijh1vh1r:11,minim:[4,7,9,15,20],minimum:[1,7,9,10,11,13,16],mit:16,mittwoch:16,mkdir:16,mlcvh347f:11,mmugcludfr:11,more:[0,1,5,9,15],mp4:[1,13],mpattf5tqt:11,mrenz0blhc:11,multi:14,multiindex:1,multipl:[10,12,20],must:[1,5,10,13],mydata:10,mydb:1,n0re5ckgsa:11,nach:16,name:[0,1,3,4,5,9,11,13,16,20],nan:[1,5,11],nat:[1,11,15],nbjerfbs9k:11,ncklr8wvrn:11,ncupmqese0:11,ndarrai:1,neben:16,necessari:8,need:[5,8],neg:[1,10],neutral:[1,16],newest:8,ngaaten4id:11,ngv7ff6ruu:11,nice:10,njv33aqxmq:11,nkbmpilgiv:11,nkgwnxzodj:11,no_schem:[9,11],non:1,none:[0,1,4,11,13,16],notabl:6,notat:13,notconformtounifiedformat:[1,5],note:[13,16,20],now:16,nth:13,nu3hauvji:11,num_fil:[4,7],num_segments_per_fil:4,num_work:[5,10,16],number:[1,4,5,13,15],numer:[1,13,14],numpi:10,nvtjdzkugz:11,nwd7gd5fka:11,nziugu3vnm:11,o4u7u83sv:11,oaasqbwwg:11,oben:16,obj:[1,5],object:[0,1,4,5,9,13,16,18],ocgbli39wu:11,odzy4afi4v:11,ofe4tphiss:11,offer:10,offici:10,often:16,ojwolqpjjq:11,omhpoggp4u:11,omiss:8,omit:13,onc:20,one:[1,4,5,9,10,13,15,18],ones:10,onli:[1,4,9,10,13],option:[1,4,13],oqydumco7c:11,order:[4,5],ore:1,organ:[1,13],origin:[1,5,13,20],ot8kzsalcx:11,other:[1,2,7,8,10,11,13,16],otherwis:[1,4],otuv5bnsfq:11,ou7s30xjva:11,ourselv:16,output:5,output_fold:5,ox3oh6ubgr:11,p_none:[1,4],pa9nu68gzi:11,packag:8,page:8,panda:[0,1,5,7,10,12,15,16,20],paper:16,papier:16,param:[10,16],paramet:[1,3,4,5],pars:14,parse_nam:16,part:[13,15,16],particip:16,pass:[1,4,20],past:16,path:[0,1,5,15,16],pd5pruq4rg:11,pdf:16,per:[1,4,9,10,13],permiss:[1,2],ph9ezp3odx:11,pho:10,phonem:10,phonet:10,pick:1,pick_column:1,pick_fil:1,pick_index:1,pick_tabl:1,pickl:2,pip:[8,19],pixel:[1,13],pj4xykcufn:11,pj9xqbytyu:11,pkl:[1,2],place:[1,16],plai:9,platz:16,pleas:8,pnzbr8yrvm:11,point:18,posit:[1,10,16],possibl:[1,4,5,8,9,12,14,20],postfix:10,ppynowxqlh:11,pq6picetbh:11,pre:[3,18],predict:11,prefix:1,prepar:16,present:16,preserv:5,principl:10,probabl:[1,4],produc:16,progress:5,progress_bar:[5,10],project:[6,8,16],prop1:11,prop2:11,properti:1,provid:[1,4,9,10,16],psu:16,pull:8,push:8,pwu3i7o42p:11,pypi:8,pystqnxaeb:11,pytest:8,python3:[8,19],python:[8,19],q4hcwd8zbr:11,q5drmrfoc4:11,q7tukbvse6:11,qbdtokvv6h:11,qbn3aukixc:11,qgd6zoshah:11,qi5sxydwol:11,qirmaofev:11,qkgd3tbimw:11,qmufz5tgdv:11,qot2dzshrf:11,qrnrr6pdmd:11,qseienbdnu:11,qtsjahvdrd:11,qugfyz7jtj:11,qytf5ywu0o:11,rais:[1,3,5],random:4,randomli:1,rang:[1,4,9],rate:[1,4,11,12,13,16],rater1:10,rater2:10,rater:[2,9,11,12,14,16],rater_id:[1,4,9,10,11,13,16],raterid:13,ratertyp:[1,10,13],read:[5,16],read_csv:[9,16],readabl:10,real:[9,16],recommend:10,reconstruct:5,record:[14,16],redund:10,redundantargumenterror:1,refer:[1,15,18],referenc:[1,12],regex:16,rel:[5,15],releas:6,reli:1,rememb:16,remov:1,rep1:16,rep:16,repetit:16,replac:[1,16],repositori:8,repres:[1,13,18],request:[8,16,20],requir:[8,9],research:[2,16],resolut:[1,13],respect:1,restrict:[1,2],result:[0,7,9],rgb:[1,13],role:10,root:[1,4,5],row:[1,11,12],rrey80izy8:11,rst:8,rtol03rbh1:11,rule:1,run:[14,19],run_task:[10,16],runter:16,runtimeerror:1,rwng20efxn:11,ryozp1fgq7:11,sac76zzeei:11,sad:16,sagen:16,same:[1,10,20],sampl:[1,4,13],sample_gener:4,sampling_r:[1,4,10,11,13,16],satz:16,save:[1,5,16,18],sayxrveap3:11,scalar:1,scheme:[2,4,7,9,11,12,16],scheme_id:[1,4,9,10,11,13,16],schemeid:13,schwarz:16,se462:16,search:14,second:[1,13],see:[1,5,10,13,16],segment:[1,2,4,5,7,9,10,11,12,14],segmented_index:15,sein:16,select:1,self:1,semant:6,sentenc:16,separ:[5,10],sequenc:[1,3,4,5],seri:[0,1,5,15,20],set:[1,4,9,10,13,15,16],set_level:1,sever:[10,18,20],should:[1,4,8,10],show:[5,9,12],shutil:16,sich:16,sie:16,sieben:16,silb:16,simpl:14,simpli:[7,8],sind:16,singl:10,situat:16,six:16,size:1,sj0gcd2yji:11,sketch:12,slow:1,small:[4,16],snje7vutxw:11,some:[7,10],sort:16,sourc:[1,2,3,4,5,8,9,10,11,13,14,19],soweit:16,spawn:5,speaker1:10,speaker2:10,speaker:[1,12,14,16,20],speaker_map:16,special:10,specif:[0,1,2,4,5,9,12,15,18],specifi:[1,14],speech:16,speecha:16,speed:1,sphinx:8,spk1:20,spk2:20,spk3:20,split:[2,4,9,11],split_id:[1,4,9,10,11,13],splitid:13,splittyp:[1,9,10,13],spoken:16,squeez:16,src:16,src_dir:16,srnp9kksmd:11,srqnhzsxnu:11,stai:8,stamp:4,standard:11,start:[1,2,5,7,11,15,18],state:16,statist:14,stehen:16,step:8,still:8,storag:[1,2],storage_format:1,store:[1,4,7,9,10,13,14,18],stpexarwfj:11,str:[1,2,3,4,5,10,11,16],str_len:1,string:[0,1,2,3,5,9,11,20],stringio:[5,9],structur:5,stunden:16,sub:[1,4,9],subset:1,support:5,sure:16,sw52wuwxe9:11,t2mikds7t6:11,tabl:[0,2,4,5,9,11,12,20],table_ex:1,table_id:[1,4,7,10,12],table_str:1,tableid:13,tablestorageformat:1,tabletyp:[1,13],tag:8,take:[10,11],task_func:[10,16],te9je5xmvd:11,technic:16,ten:16,test:[0,1,2,7,10,11,13,15,20],text:10,thei:16,them:[14,20],thi:[5,6,7,8,9,10,13,16,20],thing:8,those:8,thousand:10,thread:5,three:9,through:1,time:[1,2,4,9,10,11,15,16],timedelta64:1,timedelta:[1,4,10],timestamp:1,tisch:16,tmgrhzynbi:11,to_i:16,to_pandas_dtyp:1,to_replac:16,to_timedelta:[10,16],togeth:18,toinxfrakz:11,took:16,tool:14,tpl1dfgnug:11,tqjfaucluj:11,train:[1,2,9,10,11],transcrib:20,transcript:[10,16,20],transcription_map:16,trgping7wr:11,trinken:16,truth:2,tt9sxn30b6:11,tum9nzzhyr:11,tuo0zgnanm:11,tupl:4,tvhycqpqjd:11,two:[1,7,10,15],txt:[8,16],tyk41imuwi:11,type:[1,2,3,4,5,7,9,10,11,13,15,16],typic:13,u9zasgdsyn:11,uadjgx8ibb:11,udk3nselmx:11,udkk0j9htp:11,ukhcyz0lbz:11,ukhxjtklra:11,um41x57smm:11,umsqu1qrhx:11,umtxmee6ia:11,und:16,under:1,underli:1,understood:14,unexpect:3,unifi:14,union:[1,4,5],uniqu:[10,13],unit:[4,9,10,11,16],unittest:[4,9,11],univers:[14,16],unknown:[1,3],unrestrict:[1,2,9,10,11,16],unter:16,until:13,updat:8,uqasribqu3:11,url:13,urllib:16,urlretriev:16,usag:[1,9,10,11,13,16,18],use:[1,4,8,9,10,13,14,16,20],usecol:16,used:[1,4,10,13,20],using:[8,9,10],usual:1,util:[9,16],utt01:9,utt02:9,utt03:9,utter:16,uwae8hbram:11,uwhdhn7ce:11,uy7cncui9x:11,valenc:10,valid:[1,3,8],valid_valu:3,valu:[1,3,4,5,13,15,16,20],valueerror:[1,5],values_dict:15,values_list:15,vccxxyd62i:11,vdhnbi72hw:11,veri:10,version:[8,10,16],video:[1,2,12,13,18],video_channel:[1,11,13],video_depth:[1,11,13],video_fp:[1,11,13],video_resolut:[1,11,13],viewdoc:16,virtual:19,virtualenv:[8,19],virut:8,vluagfz4t:11,vmisgwghsd:11,vote:[2,10],vt9ukg0wop:11,vtabvmbkqh:11,vzmynx2opq:11,wafnijbr2o:11,wai:[8,14,18],want:[7,9],wav:[0,1,4,7,9,10,11,13,16,20],wbdysoyib:11,wcu9vwgryi:11,wcvosiclt5:11,webcam:11,webpag:13,websit:1,wegbringen:16,well:[4,16],were:16,what:13,when:[3,10,13,16],where:[4,13],which:[0,1,7,9,10,12,16],who:13,whole:[1,7,14,15],width:13,wieder:16,wir:16,wird:16,within:[5,10],without:[1,4,9],wjwacxex8k:11,wochenenden:16,word:[10,20],work:20,worth:10,would:[9,16],wpxylm4nck:11,write:[9,10],written:[12,18],wrn11v6doh:11,www:1,wyjxylwdeh:11,x1srxpen2i:11,x5eckksudc:11,xa0:16,xdrqmqquzd:11,xfcwl86gxl:11,xj3oy1qsyi:11,xlcqhitfv1:11,xm6i06qym6:11,xrwirkpghc:11,xry1kol88m:11,xvw75jphrf:11,xxwhleqxar:11,xxx:10,yaml:[1,11,12,16,18],ye2hdrjqy6:11,yes:13,yet:15,yield:16,yj9caiq8mm:11,yjchqiugok:11,yjk7pmps0g:11,ykydrmywbo:11,yl6ysmeex4:11,ynyvhhjs8k:11,yo5abzoaba:11,you:[0,1,4,7,8,9,10,16],your:[1,7,8,9],yp232gzdek:11,yqapz5mkgp:11,yumnuems66:11,yvf1wtsewc:11,z6abiqxetw:11,zaniwrvdd7:11,ze1xtrr5zi:11,zgzxrrq1tp:11,zip:16,zjz3ofgn5c:11,zlbzylgxml:11,znfqimdxss:11,zspmlcwv03:11,ztb2cwwzjp:11,zttfe3xjrl:11,zv8mglmpyt:11},titles:["Working with a database","audformat","audformat.define","audformat.errors","audformat.testing","audformat.utils","Changelog","Combine tables","Contributing","Create a database","Conventions","Example","Database","Header","Introduction","Tables","Emodb example","Index","audformat","Installation","Map scheme labels"],titleterms:{"2020":6,"new":8,Use:9,access:0,add_tabl:4,annot:[10,16],audformat:[1,2,3,4,5,12,13,15,16,18],audio:9,badiderror:3,badtypeerror:3,badvalueerror:3,build:8,chang:0,changelog:6,column:[1,13],combin:7,concat:5,confid:10,contribut:8,convent:10,creat:[8,9,16],create_audio_fil:4,create_db:4,csv:9,data:[0,10],databas:[0,1,9,10,12,13,16],datatyp:2,defin:2,develop:8,disk:[12,16],document:8,durat:10,emodb:16,error:3,exampl:[11,13,16],exist:9,file:[0,9,10],filewis:15,filewise_index:1,gather:16,gender:2,get:16,gold:10,hard:12,header:[13,16],implement:[13,15],index:17,index_typ:1,indexfield:2,indextyp:2,inform:10,inspect:16,instal:[8,19],introduct:14,label:20,map:20,map_languag:5,media:[1,13],mediatyp:2,metadata:16,minim:13,name:10,part:12,rater:[1,10,13],ratertyp:2,read_csv:5,referenc:0,releas:8,run:8,scheme:[1,10,13,20],segment:15,segmented_index:1,sourc:16,speaker:10,split:[1,10,13],splittyp:2,standard:10,store:[12,16],tabl:[1,7,10,13,15,16],tablestorageformat:2,tempor:10,test:[4,8,9],to_filewise_index:5,to_segmented_index:5,usag:2,util:5,valu:10,version:6,work:0}})