from __future__ import print_function
import mdtraj as mdt
import numpy as np
import collections
from collections import defaultdict
import pandas as pd
from contact_map import ContactMap, ContactFrequency, ContactDifference, ResidueContactConcurrence, plot_concurrence
import pickle
import warnings

warnings.filterwarnings("ignore", message="top= kwargs ignored since this file parser does not support it")

# Input parameters. Needed to find and name the files.

Protein = 'DNMT3A'      # Name of the protein. 
Variant = 'WT'          # WT: Wild Type or MT: Mutant.
sim_time = '25ns'       # simulation time of each replicate. usually something like: '100ns'
replicates = 1          # number of replciates one wants to analyse in the current folder 
cutoff = 0.35           # size of the sphere in nm used to calculate the contacts in.

# load the input trajectories, join and superpose them so that the become one big trajectory to analyse.

traj_dict = {}
traj_list = []
for i in range(replicates):
    folder = '/path-to-trajectory/production_DNMT3A-{}_{}_{}.h5'.format(Variant, sim_time, i+1)
    print(folder)
    traj_dict[i+1]=mdt.load(folder)
for key in traj_dict:
    traj_list.append(traj_dict[key])
traj_pre = mdt.join(traj_list,check_topology=True, discard_overlapping_frames=True)
traj = traj_pre.superpose(traj_pre,frame=0,parallel=True)
print('traj:', traj)
print('Trajectories successfully loaded, joined and superposed')
topology = traj.topology

# select interaction partners

# WT: 17534 atoms, 1028 residues
# MT: 17510 atoms, 1027 residues
# since DNMT3A/3L consists out of 4 subunits: two times DNMT3A and two time DNMT3L, the subunits are enumerated: DNMT3A-1, DNMT3A-2, DNMT3L-1, DNMT3L-2
# the two DNA strands and SAM molecules are also enumerated; DNA-1, DNA-2, SAM-1, SAM-2

# for DNMT3A-MT -> DNMT3A-1 has 284 residues
# chain ids: 0:DNMT3A-1, 1:DNMT3A-B, 2:DNMT3L-1, 3:DNMT3L-2, 4:DNA-1, 5:DNA-2, 6:SAM-1, 7:SAM-2
# atoms: DNMT3A-1:0-4603, DNMT3A-2:40604-9207, DNMT3L-1:9208-12525, DNMT3L-B:12526-15843, DNA-1:15844-16639, DNA-2:16640-17435, SAM-1:17436-17462, SAM-2:17485-17511
# residues: DNMT3A-1:0-284, DNMT3A-2:285-569, DNMT3L-1:570-772, DNMT3L-2:773-975, DNA-1:976-1000, DNA-2:1001-1025, SAM-1:1026, SAM-2:1027
# residues total: DNMT3A-1: 285, DNMT3A-2:285, DNMT3L-1:203, DNMT3L-2:203, DNA-1:25, DNA-2:25, SAM-1:1, SAM-2:1

subunit_3A = topology.select('chainid 0 and element != "H"') # all heavy atoms of DNMT3A-1 are selcted here. Hydrogens are excluded.
interaction_partner = topology.select('chainid 1 or chainid 2 or chainid 4 or chainid 5 and element != "H"') # DNMT3A-2, DNMT3L-1, DNA-1, DNA-2

print('Calculating Frequency of contacts')
contacts = ContactFrequency(traj, query=subunit_3A, haystack=interaction_partner, cutoff=cutoff) # calculation of the contacts. Each possible contact between the selected molecules is given a value between 0 and 1 depending on its observed frequence. 1 = 100% of the simulaiton time, the two residue had atoms in the 3.5A sphere of the other residue.
x = contacts.residue_contacts.sparse_matrix.toarray() # creates a matrix with the calcualted frequencies.
df = pd.DataFrame(x) # converts the matrix into a pandas dataframe

# for the mutant, the dataframe needs to be modified for subsequent calcualtions
if Variant == 'MT':
    x1 = [0] * 1027
    x2 = [0] * 1028
    df.insert(0, column='A628', value=x1) # adding a dummy column
    df.loc[-1] = x2  # adding a dummy row
    df.index = df.index + 1  # shifting index
    df.sort_index(inplace=True)

# since every possible contact is listed in the inital array, we can drop columns and rows that are not relevant
df.drop(df.iloc[:, 285:1028], inplace = True, axis = 1) # on x-axis only DNMT3A-1
df.drop(df.index[773:976], inplace=True, axis = 0)   # exclude DNMT3L-2
df.drop(df.index[0:285], inplace=True, axis = 0)     # exclude DNMT3A-1
df.drop(df.index[538:541], inplace=True, axis = 0)   # exclude SAM-1, SAM-2

# name the columns and rows with the respective residues
if Variant == 'WT':
    df.index = ['A628','E629','K630','R631','K632','P633','I634','R635','V636','L637','S638','L639','F640','D641','G642','I643','A644','T645','G646','L647','L648','V649','L650','K651','D652','L653','G654','I655','Q656','V657','D658','R659','Y660','I661','A662','S663','E664','V665','C666','E667','D668','S669','I670','T671','V672','G673','M674','V675','R676','H677','Q678','G679','K680','I681','M682','Y683','V684','G685','D686','V687','R688','S689','V690','T691','Q692','K693','H694','I695','Q696','E697','W698','G699','P700','F701','D702','L703','V704','I705','G706','G707','S708','P709','C710','N711','D712','L713','S714','I715','V716','N717','P718','A719','R720','K721','G722','L723','Y724','E725','G726','T727','G728','R729','L730','F731','F732','E733','F734','Y735','R736','L737','L738','H739','D740','A741','R742','P743','K744','E745','G746','D747','D748','R749','P750','F751','F752','W753','L754','F755','E756','N757','V758','V759','A760','M761','G762','V763','S764','D765','K766','R767','D768','I769','S770','R771','F772','L773','E774','S775','N776','P777','V778','M779','I780','D781','A782','K783','E784','V785','S786','A787','A788','H789','R790','A791','R792','Y793','F794','W795','G796','N797','L798','P799','G800','M801','N802','R803','P804','L805','A806','S807','T808','V809','N810','D811','K812','L813','E814','L815','Q816','E817','C818','L819','E820','H821','G822','R823','I824','A825','K826','F827','S828','K829','V830','R831','T832','I833','T834','T835','R836','S837','N838','S839','I840','K841','Q842','G843','K844','D845','Q846','H847','F848','P849','V850','F851','M852','N853','E854','K855','E856','D857','I858','L859','W860','C861','T862','E863','M864','E865','R866','V867','F868','G869','F870','P871','V872','H873','Y874','T875','D876','V877','S878','N879','M880','S881','R882','L883','A884','R885','Q886','R887','L888','L889','G890','R891','S892','W893','S894','V895','P896','V897','I898','R899','H900','L901','F902','A903','P904','L905','K906','E907','Y908','F909','A910','C911','V912', 'M178','F179','E180','T181','V182','P183','V184','W185','R186','R187','Q188','P189','V190','R191','V192','L193','S194','L195','F196','E197','D198','I199','K200','K201','E202','L203','T204','S205','L206','G207','F208','L209','E210','S211','G212','S213','D214','P215','G216','Q217','L218','K219','H220','V221','V222','D223','V224','T225','D226','T227','V228','R229','K230','D231','V232','E233','E234','W235','G236','P237','F238','D239','L240','V241','Y242','G243','A244','T245','P246','P247','L248','G249','H250','T251','C252','D253','R254','P255','P256','S257','W258','Y259','L260','F261','Q262','F263','H264','R265','L266','L267','Q268','Y269','A270','R271','P272','K273','P274','G275','S276','P277','R278','P279','F280','F281','W282','M283','F284','V285','D286','N287','L288','V289','L290','N291','K292','E293','D294','L295','D296','V297','A298','S299','R300','F301','L302','E303','M304','E305','P306','V307','T308','I309','P310','D311','V312','H313','G314','G315','S316','L317','Q318','N319','A320','V321','R322','V323','W324','S325','N326','I327','P328','A329','I330','R331','S332','R333','H334','W335','A336','L337','V338','S339','E340','E341','E342','L343','S344','L345','L346','A347','Q348','N349','K350','Q351','S352','S353','K354','L355','A356','A357','K358','W359','P360','T361','K362','L363','V364','K365','N366','C367','F368','L369','P370','L371','R372','E373','Y374','F375','K376','Y377','F378','S379','T380', 'C423','A424','T425','G426','C427','G428','A429','T430','C431','T432','A433','A434','T435','T436','A437','G438','A439','T440','C441','G442','C443','A444','T445','G446','G447', 'C423','A424','T425','G426','C427','G428','A429','T430','C431','T432','A433','A434','T435','T436','A437','G438','A439','T440','C441','G442','C443','A444','T445','G446','G447']
    df.columns = ['A628','E629','K630','R631','K632','P633','I634','R635','V636','L637','S638','L639','F640','D641','G642','I643','A644','T645','G646','L647','L648','V649','L650','K651','D652','L653','G654','I655','Q656','V657','D658','R659','Y660','I661','A662','S663','E664','V665','C666','E667','D668','S669','I670','T671','V672','G673','M674','V675','R676','H677','Q678','G679','K680','I681','M682','Y683','V684','G685','D686','V687','R688','S689','V690','T691','Q692','K693','H694','I695','Q696','E697','W698','G699','P700','F701','D702','L703','V704','I705','G706','G707','S708','P709','C710','N711','D712','L713','S714','I715','V716','N717','P718','A719','R720','K721','G722','L723','Y724','E725','G726','T727','G728','R729','L730','F731','F732','E733','F734','Y735','R736','L737','L738','H739','D740','A741','R742','P743','K744','E745','G746','D747','D748','R749','P750','F751','F752','W753','L754','F755','E756','N757','V758','V759','A760','M761','G762','V763','S764','D765','K766','R767','D768','I769','S770','R771','F772','L773','E774','S775','N776','P777','V778','M779','I780','D781','A782','K783','E784','V785','S786','A787','A788','H789','R790','A791','R792','Y793','F794','W795','G796','N797','L798','P799','G800','M801','N802','R803','P804','L805','A806','S807','T808','V809','N810','D811','K812','L813','E814','L815','Q816','E817','C818','L819','E820','H821','G822','R823','I824','A825','K826','F827','S828','K829','V830','R831','T832','I833','T834','T835','R836','S837','N838','S839','I840','K841','Q842','G843','K844','D845','Q846','H847','F848','P849','V850','F851','M852','N853','E854','K855','E856','D857','I858','L859','W860','C861','T862','E863','M864','E865','R866','V867','F868','G869','F870','P871','V872','H873','Y874','T875','D876','V877','S878','N879','M880','S881','R882','L883','A884','R885','Q886','R887','L888','L889','G890','R891','S892','W893','S894','V895','P896','V897','I898','R899','H900','L901','F902','A903','P904','L905','K906','E907','Y908','F909','A910','C911','V912']

if Variant == 'MT':
    df.index = ['A628','E629','K630','R631','K632','P633','I634','R635','V636','L637','S638','L639','F640','D641','G642','I643','A644','T645','G646','L647','L648','V649','L650','K651','D652','L653','G654','I655','Q656','V657','D658','R659','Y660','I661','A662','S663','E664','V665','C666','E667','D668','S669','I670','T671','V672','G673','M674','V675','R676','H677','Q678','G679','K680','I681','M682','Y683','V684','G685','D686','V687','R688','S689','V690','T691','Q692','K693','H694','I695','Q696','E697','W698','G699','P700','F701','D702','L703','V704','I705','G706','G707','S708','P709','C710','N711','D712','L713','S714','I715','V716','N717','P718','A719','R720','K721','G722','L723','Y724','E725','G726','T727','G728','R729','L730','F731','F732','E733','F734','Y735','R736','L737','L738','H739','D740','A741','R742','P743','K744','E745','G746','D747','D748','R749','P750','F751','F752','W753','L754','F755','E756','N757','V758','V759','A760','M761','G762','V763','S764','D765','K766','R767','D768','I769','S770','R771','F772','L773','E774','S775','N776','P777','V778','M779','I780','D781','A782','K783','E784','V785','S786','A787','A788','H789','R790','A791','R792','Y793','F794','W795','G796','N797','L798','P799','G800','M801','N802','R803','P804','L805','A806','S807','T808','V809','N810','D811','K812','L813','E814','L815','Q816','E817','C818','L819','E820','H821','G822','R823','I824','A825','K826','F827','S828','K829','V830','R831','T832','I833','T834','T835','R836','S837','N838','S839','I840','K841','Q842','G843','K844','D845','Q846','H847','F848','P849','V850','F851','M852','N853','E854','K855','E856','D857','I858','L859','W860','C861','T862','E863','M864','E865','R866','V867','F868','G869','F870','P871','V872','H873','Y874','T875','D876','V877','S878','N879','M880','S881','H882','L883','A884','R885','Q886','R887','L888','L889','G890','R891','S892','W893','S894','V895','P896','V897','I898','R899','H900','L901','F902','A903','P904','L905','K906','E907','Y908','F909','A910','C911','V912', 'M178','F179','E180','T181','V182','P183','V184','W185','R186','R187','Q188','P189','V190','R191','V192','L193','S194','L195','F196','E197','D198','I199','K200','K201','E202','L203','T204','S205','L206','G207','F208','L209','E210','S211','G212','S213','D214','P215','G216','Q217','L218','K219','H220','V221','V222','D223','V224','T225','D226','T227','V228','R229','K230','D231','V232','E233','E234','W235','G236','P237','F238','D239','L240','V241','Y242','G243','A244','T245','P246','P247','L248','G249','H250','T251','C252','D253','R254','P255','P256','S257','W258','Y259','L260','F261','Q262','F263','H264','R265','L266','L267','Q268','Y269','A270','R271','P272','K273','P274','G275','S276','P277','R278','P279','F280','F281','W282','M283','F284','V285','D286','N287','L288','V289','L290','N291','K292','E293','D294','L295','D296','V297','A298','S299','R300','F301','L302','E303','M304','E305','P306','V307','T308','I309','P310','D311','V312','H313','G314','G315','S316','L317','Q318','N319','A320','V321','R322','V323','W324','S325','N326','I327','P328','A329','I330','R331','S332','R333','H334','W335','A336','L337','V338','S339','E340','E341','E342','L343','S344','L345','L346','A347','Q348','N349','K350','Q351','S352','S353','K354','L355','A356','A357','K358','W359','P360','T361','K362','L363','V364','K365','N366','C367','F368','L369','P370','L371','R372','E373','Y374','F375','K376','Y377','F378','S379','T380', 'C423','A424','T425','G426','C427','G428','A429','T430','C431','T432','A433','A434','T435','T436','A437','G438','A439','T440','C441','G442','C443','A444','T445','G446','G447', 'C423','A424','T425','G426','C427','G428','A429','T430','C431','T432','A433','A434','T435','T436','A437','G438','A439','T440','C441','G442','C443','A444','T445','G446','G447']
    df.columns = ['A628','E629','K630','R631','K632','P633','I634','R635','V636','L637','S638','L639','F640','D641','G642','I643','A644','T645','G646','L647','L648','V649','L650','K651','D652','L653','G654','I655','Q656','V657','D658','R659','Y660','I661','A662','S663','E664','V665','C666','E667','D668','S669','I670','T671','V672','G673','M674','V675','R676','H677','Q678','G679','K680','I681','M682','Y683','V684','G685','D686','V687','R688','S689','V690','T691','Q692','K693','H694','I695','Q696','E697','W698','G699','P700','F701','D702','L703','V704','I705','G706','G707','S708','P709','C710','N711','D712','L713','S714','I715','V716','N717','P718','A719','R720','K721','G722','L723','Y724','E725','G726','T727','G728','R729','L730','F731','F732','E733','F734','Y735','R736','L737','L738','H739','D740','A741','R742','P743','K744','E745','G746','D747','D748','R749','P750','F751','F752','W753','L754','F755','E756','N757','V758','V759','A760','M761','G762','V763','S764','D765','K766','R767','D768','I769','S770','R771','F772','L773','E774','S775','N776','P777','V778','M779','I780','D781','A782','K783','E784','V785','S786','A787','A788','H789','R790','A791','R792','Y793','F794','W795','G796','N797','L798','P799','G800','M801','N802','R803','P804','L805','A806','S807','T808','V809','N810','D811','K812','L813','E814','L815','Q816','E817','C818','L819','E820','H821','G822','R823','I824','A825','K826','F827','S828','K829','V830','R831','T832','I833','T834','T835','R836','S837','N838','S839','I840','K841','Q842','G843','K844','D845','Q846','H847','F848','P849','V850','F851','M852','N853','E854','K855','E856','D857','I858','L859','W860','C861','T862','E863','M864','E865','R866','V867','F868','G869','F870','P871','V872','H873','Y874','T875','D876','V877','S878','N879','M880','S881','H882','L883','A884','R885','Q886','R887','L888','L889','G890','R891','S892','W893','S894','V895','P896','V897','I898','R899','H900','L901','F902','A903','P904','L905','K906','E907','Y908','F909','A910','C911','V912']

# safe the calcualted contact frequencies either in in pickle or as an excel sheet. The pickle file is used for further calcualtions as in "DNMT3A-Contact_Maps_Analysis.py" or "DNMT3A-Contact_Maps_Statistics.py"
number_replicates = str(replicates)
contact_cutoff_str = str(cutoff).replace('.', '_')
df.to_pickle('DNMT3A-{}-{}x{}_cutoff_{}.pkl'.format(Variant, number_replicates, sim_time, contact_cutoff_str))
df.to_excel('DNMT3A-{}-{}x{}_cutoff_{}.xlsx'.format(Variant, number_replicates, sim_time, contact_cutoff_str))
