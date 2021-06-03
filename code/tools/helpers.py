import os,re
import numpy as np
from functools import partial
from pathlib import Path
from Levenshtein import distance
from collections import defaultdict
import pandas as pd
from collections import deque
import random
# helper function
get_doc_name = lambda x: x.split("/")[-1].split(".")[0]
get_doc_page = lambda x: int(get_doc_name(x).split("_")[-1])
get_folder_name = lambda x: x.split("/")[-3]
#strip_anno_idx = lambda x: x.split('[')[0]
strip_anno_idx = lambda x: re.sub('\[[0-9]+?\]','',x)

def process_line(line):
	line = re.sub(r"([\.\—,?!\&\(\)\[\]\{\}])", r' \1 ',line)
	line = re.sub(r"\\", '/',line)
	line = [t for t in line.split() if t.strip()]
	return line


def checkpath(output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

get_tag_and_idx = lambda x,y : (x.split('[')[0], y.findall(x))
#strip_anno_idx = lambda x: re.sub(r'\[[0-9]+?\]','',x)


def range_to_pagenumbers(editions_range):
    editions_pages = defaultdict(list)
    for y, y_l in editions_range.items():
        for r in y_l:
            editions_pages[y].extend(list(range(*r)))
    return editions_pages

def stratified_random_subsample(editions_range,size=4):
    editions_pages = defaultdict(list)
    for y, y_l in editions_range.items():
        for r in y_l:
            pages = list(range(*r))
            random.shuffle(pages)
            editions_pages[y].extend(pages[:size])
    return editions_pages


def replace_tokens_with_closest(expression,masterlist):
    if not isinstance(expression,str): return None

    if expression in masterlist: return expression

    part_funct = partial(distance,expression.rstrip('*').strip())
    distances = list(map(part_funct,masterlist))
    dist, pos = np.min(distances), np.argmin(distances)

    if dist <= 3:
        return masterlist[pos]
    else:
        return '@@' + masterlist[pos]


def replace_subtokens_with_closest(expression,masterlist):
    if isinstance(expression,str):
        expressions = [e for e in re.split('\s|(?<!non)-|(?:<sep>)',expression.strip()) if e]
        
        return '; '.join([replace_tokens_with_closest(e, masterlist) for e in expressions])
    return None

def clean_df(df):

    def clean_district_name(x):
        x_clean = ' '.join(str(x).strip().lstrip('(').split(')')[:-1]).strip()
        if x_clean:
            return x_clean
        return x


    labels = open('/deezy_datadrive/kaspar-playground/Living-with-Machines-code/sources-lab-mro/npd_pipeline/npd_final/newspaper_metadata/political_labels.txt').read().strip().split('\n')
    districts = [d.lower() for d in open('/deezy_datadrive/kaspar-playground/Living-with-Machines-code/sources-lab-mro/npd_pipeline/npd_final/newspaper_metadata/district_names.txt').read().strip().split('\n')]
    
    df[f'S-TITLE'] = df[f'S-TITLE'].apply(lambda x: str(x).replace('<SEP>',' '))
    df[f'S-TITLE'] = df[f'S-TITLE'].apply(lambda x: str(x).replace('<CON>',' '))
    df[f'S-TITLE'] = df[f'S-TITLE'].apply(lambda x: str(x).replace('@',''))

    df[f'S-POL_corr'] = df['S-POL'].str.lower().apply(replace_subtokens_with_closest,masterlist=labels)
    df[f'DISTRICT_corr'] = df['DISTRICT'].str.lower().apply(
                                                    clean_district_name).apply(
                                                        replace_tokens_with_closest,masterlist=districts)

    df = df[["id", "S-TITLE",'S-POL','S-POL_corr',"DISTRICT","DISTRICT_corr","S-PRICE","D-EST","D-PUB","E-LOC","E-ORG","E-PER","S-TITLE-ALT","TEXT",'DISTRICT_DESCRIPTION']]
    
    #for col in df.columns:
    #    df[col] = df[col].apply(lambda x: x.replace('<CON>',' '))
    return df

def clean_corrected_csv_depr(path):
    """kind of a catch-all function that ensures final data 
    is in the right shape for for linking and further analysis
    """

    countydict =  {
    'london':'',
    '':'',
    'ayrshare': 'ayrshire',
    'banffshire':'banffrshire',
    'bucks':'buckinghamshire',
    'carnarvon':'carnarvonshire',
    'cheshunt . ( herts':'hertfordshire',
    'clackmannan':'clackmannanshire',
    'crowle - ( lincolnshire':'lincolnshire',
    'devon':'devonshire',
    'dorset':'dorsetshire',
    'dunmfriesshire':'dumfriesshire',
    'flint':'flintshire',
    'glamorgan':'glamorganshire',
    'goloucestershire':'gloucestershire',
    'granton-on-spey - ( elginshire':'elginshire',
    'grantown - ( elginshire':'elginshire',
    'handsworth & smethwick . ( staffordshire':'staffordshire',
    'hants':'hampshire',
    'hertford':'hertfordshire',
    'herts':'hertfordshire',
    'irlam - ( lancashire':'lancashire',
    'kirriemuir - ( forfarshire':'forfarshire',
    'lampeter - ( cardiganshire':'cardiganshire',
    'laurencekirk . ( kincardineshire':'kincardineshire',
    'leicestersh':'leicestershire',
    'monmouthsire':'monmouthshire',
    'notltinghamshire':'nottinghamshire',
    'notts':'nottinghamshire',
    'peeblesshire':'peebleshire',
    'radcliffe ( lancashire':'lancashire',
    'somerset':'somersetshire',
    'stafordshire':'staffordshire',
    'stafss':'staffordshire',
    'sufolk':'suffolk',
    'surrey . ) \t':'surrey',
    'swinton and pendlebury . ( lancashire':'lancashire',
    'uddingston - ( lanarkshire':'lanarkshire',
    'westmorland':'westmoreland',
    'wigtonnshire':'wigtownshire',
    'willtshire':'wiltshire',
    'wilts':'wiltshire',
    'wokingham ( berkshire':'berkshire',
    'worcestershntre':'worcestershire',
    'yortkshire':'yorkshire',
    'yortshire':'yorkshire'}

    get_county = lambda x: str(x).split('— ')[-1].strip('() ').lower().strip(". )")
    get_count_dict = lambda x: countydict.get(x,x)

    _ , name = path.parent, path.stem
    df = pd.read_csv(path)
    df['DISTRICT'] = df['DISTRICT_corr'].str.replace('@@','')
    df['DISTRICT'] = df['DISTRICT'].str.lower()
    df['COUNTY'] = df.DISTRICT.apply(get_county)
    df['COUNTY'] = df.COUNTY.apply(get_count_dict)
    df['S-POL'] = df['S-POL_corr'].str.replace('@@','')
    year = name.split('_')[1]
    df['id'] = df.apply(lambda x: f'MPD_{year}_{int(x.name)}',axis=1)
    df = df[["id", "S-TITLE",'S-POL','CATEGORY',"DISTRICT",'COUNTY',"S-PRICE","D-EST","D-PUB","E-LOC","E-ORG","E-PER","S-TITLE-ALT","TEXT",'DISTRICT_DESCRIPTION']]
    
    out_folder = Path('csv_dump_final')
    out_folder.mkdir(exist_ok=True)

    df.to_csv(out_folder / (name[:8] + '.csv'))


def clean_corrected_csv(path,district_names_csv='./newspaper_metadata/district_names.csv'):
    """kind of a catch-all function that ensures final data 
    is in the right shape for for linking and further analysis
    """
    _ , name = path.parent, path.stem
    #print(name)
    get_from_dict_soft = lambda x, dict_sel: dict_sel.get(x,x)
    only_char = lambda x: ''.join(i for i in x if i.isalpha())
    get_from_dict = lambda x, dict_sel: dict_sel[x]
    replace_dict = {'swinton and pendlebury . ( lancashire . )':'swinton and pendlebury . — ( lancashire . )',
                    "cowes . — (isle of wight . )":'cowes . — ( isle of wight . )',
                    'stamford . — ( launcolnshire . ) ':'stamford . — ( lincolnshire . )',
                    'uxbridge . — ( midadtlesex . )':"uxbridge . — ( middlesex . )",
                    "jedrburgh . — ( roxburghshire . )":'jedburgh . — ( roxburghshire . )',
                    'isle of man':'isle of man . — (isle of man . )',
                    'jersey':'jersey . — ( jersey )',
                    'guernsey .':'guernsey . — ( guernsey )',
                    'welshpool . — ( montgomershire  . )':'welshpool . — ( montgomeryshire . )',
                    'monhouth . — ( monmouthshire . )':'monmouth . — ( monmouthshire . )',
                    'alton . — ( hampshare . )':'alton . — ( hampshire . )',
                    'stamford . — ( launcolnshire . )':'stamford . — ( lincolnshire . )',
                    'loughrea . — ( in the province of connaught and county galway . )':'loughrea . — ( in the province of connaught and county galway . )'
                    }

    df_district  = pd.read_csv(district_names_csv,index_col=0)
    df_district['original'] = df_district.original.str.lower()
    df_district['original'] = df_district.original.apply(only_char)
    df_district.set_index('original',drop=True,inplace=True)
    district_dict = df_district[['district']].to_dict()['district']
    county_dict = df_district[['county']].to_dict()['county']
    
    #get_county = lambda x: str(x).split('— ')[-1].strip('() ').lower().strip(". )")
    #get_from_dict = lambda x, dict_sel: dict_sel.get(x,x)
    
    

    df = pd.read_csv(path)

    

    df['DISTRICT'] = df['DISTRICT_corr'].str.replace('@@','').str.rstrip()
    df['DISTRICT'] = df['DISTRICT'].str.lower()
    df['DISTRICT'] = df.DISTRICT.apply(get_from_dict_soft, dict_sel=replace_dict)
    
    df['DISTRICT_TEMP'] = df['DISTRICT'].apply(only_char)

    
    #df['DISTRICT'] = df['DISTRICT'].apply(str.strip)
    
    #df['COUNTY'] = df.DISTRICT.apply(get_county)
    df['DISTRICT_PUB'] = df.DISTRICT_TEMP.apply(get_from_dict, dict_sel=district_dict)
    df['COUNTY'] = df.DISTRICT_TEMP.apply(get_from_dict, dict_sel=county_dict)
    df['S-POL'] = df['S-POL_corr'].str.replace('@@','')
    year = name.split('_')[1]
    df['id'] = df.apply(lambda x: f'MPD_{year}_{int(x.name)}',axis=1)
    df = df[["id", "S-TITLE",'S-POL','CATEGORY',"DISTRICT","DISTRICT_PUB",'COUNTY',"S-PRICE","D-EST","D-PUB","E-LOC","E-ORG","E-PER","S-TITLE-ALT","TEXT",'DISTRICT_DESCRIPTION']]
    
    out_folder = Path('csv_dump_final')
    out_folder.mkdir(exist_ok=True)

    df.to_csv(out_folder / (name[:8] + '.csv'))

def clean_all_csv(path='./csv_dump_temp/'):
    paths = list(Path(path).glob('**/*.csv'))
    deque(map(clean_corrected_csv,paths))



def insert_wikidata_ids(npd_path,district_authority_file_path,wikidata_gazetter_path):
    df_auth = pd.read_csv(district_authority_file_path, index_col=0)
    df_wiki = pd.read_csv(wikidata_gazetter_path)
    df_npd = pd.read_csv(npd_path,index_col=0)
    
    not_in_gaz = set(df_auth.wiki_id).difference(set(df_wiki.wikidata_id))
    print('Adding these to wikidata gazetteer')
    print(not_in_gaz)

    df_auth['original'] = df_auth.original.str.lower()
    df_wiki_linked = df_wiki[df_wiki.wikidata_id.isin(set(df_auth.wiki_id))]


    selected_cols = ['wikidata_id','latitude','longitude','hcounties']
    df_auth = df_auth.merge(df_wiki_linked[selected_cols], right_on='wikidata_id',left_on='wiki_id', how='left')

    df_npd_merged = df_npd.merge(df_auth,left_on='DISTRICT',right_on='original', how='left')
    print(df_npd.shape,df_npd_merged.shape)
    return df_npd_merged


