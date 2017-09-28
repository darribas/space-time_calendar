import time
from subprocess import check_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysal as ps
from pysal.contrib.viz import mapping as maps

##################
### Processing ###
##################

def process_day(p, shp):
    '''
    Process a single day file and return table indexed on area/time
    ...

    Arguments
    ---------
    p       : str
              Path to file to process
    shp     : geotable
              Table with grid geometries indexed on `cellId`

    Returns
    -------
    act     : DataFrame
              Table indexed on area/time containing activity levels
    '''
    f = p.split('/')[-1]
    print("Processing file: %s"%f)
    t0 = time.time()
    # Load up file
    names = ['gID', 'time_int', 'sms_in', \
         'sms_out', 'call_in', 'call_out', \
         'int_traf', 'cty']
    db = pd.read_table(p, names=names)
    # Shrink to boundaries
    db = db.set_index('gID')\
            .loc[shp.index, :]\
            .reset_index()
    # Aggregate by area/timestamp
    sqrd = db.groupby(['gID', 'time_int'])\
            .sum()\
            .fillna(0)\
            .drop(['cty'], axis=1)\
            .reset_index()
    sqrd['time_int'] = sqrd['time_int'].apply(dp)
    # Resample
    resampler = lambda x: x.set_index('time_int')\
                        .resample('h')\
                        .sum()
    hly = sqrd.groupby('gID')\
                .apply(resampler)\
                .drop(['gID'], axis=1)\
                .reset_index()
    hly['h'] = hly['time_int'].apply(lambda x: x.hour)
    hly = hly.set_index(['gID', 'h'])\
            .drop(['time_int'], axis=1)
    print("\tFile processed in %.2f seconds"%(time.time()-t0))
    return hly

dp = lambda t: pd.Timestamp(int(t), unit='ms')

def w_stitch_single(w_orig, t, back=0, forth=0, silent_island_warning=False):
    '''
    Generate a space-time weights object, `w`, that stacks the weights matrix
    (`w`) `t` number of times and connects each observation with its
    contemporary neighbors across `back` and `forth` steps
    
    ...

    Arguments
    ---------
    w                       : W
                              Weights matrix to be replicated over `t`
                              periods.
    t                       : int
                              Number of periods to replicate `w` over.
    back                    : int
                              [Optional. Default=0] Number of periods an
                              observation is connected backwards.
    forth                   : int
                              [Optional. Default=0] Number of periods an
                              observation is connected forwards.
    silent_island_warning   : boolean
                              Switch to turn off (default on) print statements

    Returns
    -------
    w_out                   : W
                              Resulting `ps.W` object

    Notes
    -----
    This is a more memory efficient version of `w_stitch` for the particular
    case in which the geography in every period is the same.

    The resulting `w` contains the original indices, converted to strings if
    necessary and preceded by 'X-', where X is the order of the original `W`
    object in `ws`.

    IMPORTANT: Weights are copied from the original weights object and do not
    have any further check. Make sure you do not pass standardized weights!

    Examples
    --------

    Build the weights for a standard lattice:

    >>> import pysal as ps
    >>> w = ps.lat2W(3, 3)
    >>> w.n
    9
    >>> w[0]
    {1: 1.0, 3: 1.0}
    >>>

    Let us stitch `w` over three periods without any connection:

    >>> w_stitched = ps.weights.Wsets.w_stitch_single(w, 3)

    First, we can check tha the order of the observations is created as
    estipulated:

    >>> w_stitched.id_order
    ['0-0', '0-1', '0-2', '0-3', '0-4', '0-5', '0-6', '0-7', '0-8', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '2-0', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8']

    Observation 0 in the same period has a new id but has the same neighbors
    as before:

    >>> w_stitched['0-0']
    {'0-1': 1.0, '0-3': 1.0}
 
    Now let's stitch the same sequence one period back and one period forward:

    >>> w_stitched1b1f = ps.weights.Wsets.w_stitch_single(w, 3, back=1, forth=1)

    The logic applies equally, so both the first and last time periods are
    stitched only when possible:

    >>> w_stitched1b1f['0-0']
    {'0-1': 1.0, '0-3': 1.0, '1-1': 1.0, '1-3': 1.0}
    >>> w_stitched1b1f['2-0']
    {'1-1': 1.0, '1-3': 1.0, '2-1': 1.0, '2-3': 1.0}

    But the observations in the middle period are stitched both back and
    forth:

    >>> w_stitched1b1f['1-0']
    {'0-1': 1.0, '0-3': 1.0, '1-1': 1.0, '1-3': 1.0, '2-1': 1.0, '2-3': 1.0}
    '''
    out_neigh = {}
    out_weigh = {}
    out_ids = []
    for i in range(t):
        for el in w_orig.neighbors:
            # Contemporary neighbors
            out_neigh['%i-%s'%(i, str(el))] = ['%i-%s'%(i, str(j)) \
                                            for j in w_orig.neighbors[el]]
            out_weigh['%i-%s'%(i, str(el))] = []
            out_weigh['%i-%s'%(i, str(el))] += w_orig.weights[el]
            # Backward neighbors
            for tb in range(1, back+1):
                if i-tb in range(t):
                    back_neigh = ['%i-%s'%(i-tb, str(j)) \
                                for j in w_orig.neighbors[el]]
                    back_weigh = w_orig.weights[el]
                    out_neigh['%i-%s'%(i, str(el))] += back_neigh
                    out_weigh['%i-%s'%(i, str(el))] += w_orig.weights[el]
            # Forward neighbors
            for tf in range(1, forth+1):
                if i+tf in range(t):
                    forth_neigh = ['%i-%s'%(i+tf, str(j)) \
                                for j in w_orig.neighbors[el]]
                    forth_weigh = w_orig.weights[el]
                    out_neigh['%i-%s'%(i, str(el))] += forth_neigh
                    out_weigh['%i-%s'%(i, str(el))] += forth_weigh
        wid = ['%i-%s'%(i, str(j)) for j in w_orig.id_order]
        out_ids.extend(wid)
    outW = ps.W(out_neigh, out_weigh, id_order=out_ids, \
            silent_island_warning=silent_island_warning) 
    return outW

#################
### Vis tools ###
#################

lisa_clrs = {1: '#FF0000', 2: '#66CCFF', 3: '#003399', 4: '#CD5C5C', 
            0: '#D3D3D3'}

def lisa_plot(lisa, shps, p_thr=0.05, ax=None):
    '''
    Generate a map with the output of a `lisa` object and a set of geometries
    ...

    Arguments
    ---------

    lisa    : Moran_Local
              LISA object
    shps    : GeoCol
              Series containing PySAL geometries on which the LISA is based.
              Assumed to be aligned with elements in `lisa`
    p_thr   : float
              [Optional. Default=0.05] Significance level for LISAs
    ax      : axes
              [Optional. Default=None] Axis to insert the plot into.

    Returns
    -------
    ax/None
    '''
    sign = lisa.p_sim < p_thr
    quadS = lisa.q * sign
    show = False
    if not ax:
        f, ax = plt.subplots(1, figsize=(9, 9))
    gt = pd.DataFrame({'geometry': shps, 
                    'val': pd.Series(quadS, index=shps.index)})
    #gt = gt.loc[gt['val'] != 0, :] # Turn on for only significant plotted
    colors = gt['val'].map(lisa_clrs)
    maps.geoplot(gt, facecolor=colors, ax=ax, 
            edgecolor='white', linewidth=0.1)
    if show:
        plt.show()
        return None
    else:
        return ax

def lisa_plots(gt, saveto=None):
    '''
    Wrapper to generate figure of hourly plots
    ...

    Arguments
    ---------
    gt      : geotable
              Table containing geometries and 24-hour LISA results
    saveto  : None/str
              [Optional. Default=None] Path to file to save the figure to as a
              png
    '''
    f, axs = plt.subplots(6, 4, figsize=(24, 24))
    axs = axs.flatten()
    for h, ax in zip(gt.columns, axs):
        q = gt.loc[:, ['geometry', h]]
        colors = q[h].map(lisa_clrs)
        maps.geoplot(gt, facecolor=colors, ax=ax, 
            edgecolor='white', linewidth=0.1)
        ax.set_title('H: %i'%h, fontsize=20)
        ax.axis('equal')
    f.suptitle('Repeated Cross-Sections | LISA', 
                fontsize=22)
    if saveto:
        plt.savefig(saveto)
    return plt.show()

#########################
### SaTScan interface ###
#########################

satscan = '/Users/dani/code/satscan.9.4_mac/satscan'
satscan = 'satscan/satscan_stdc++6_x86_64_64bit'

def scan(y, shp_link, id_col, sp_win=50, t_win=50, stype=1, rep=999,
        opath=None, p_thres=0.05, print_cl_info=True, weight='bag',
        tpl_link='satscan_template_normal.prm', verbose=True, sp_only=False, int_ids=True):
    '''
    Compute normal space-time statistics with SaTScan

    ...

    Arguments
    ---------
    y               : Series
                      Variable MultiIndex'ed with location and time ID,
                      respectively.
    shp_link        : str
                      Path to the shapefile for geography.
    id_col          : str
                      Column in DBF of `shp_link` with IDs to match to `y`.
    sp_win          : float
                      [Optional. Default=50] Maximum percentage of the total
                      dataset to be used for the spatial window of the scan.
    t_win           : float
                      [Optional. Default=50] Maximum percentage of the total
                      dataset to be used for the time window of the scan.
    stype           : int
                      [Optional. Default=1] Type of scan:
                        * 1: high values
                        * 2: low values
    rep             : int
                      [Optional. Default=999] Replications for obtain
                      inference.
    opath           : str
                      [Optional. Default=None] Path to set for SaTScan files
                      (do not include extension as it will create more than one).
                      If no `opath` is passed, all files created in the
                      process of running this method are temporary and thus
                      removed.
    p_thres         : float
                      [Optional. Default=0.05] Threshold for a cluster to be
                      considered significant.
    print_cl_info   : boolean
                      [Optional. Default=True] Switch to print info on
                      clusters from SaTScan
    weight          : None/str
                      [Optional. Default='bag'] Switch to use a weighted normal
                      statistic. It may take the following values:

                        * `bag`
                        * `area`
                        * `area_inv`
                        * `bag_dens`
                        * None
    tpl_link        : str
                      [Optional. Default='satscan_template_normal.prm'] Path
                      to prm template
    verbose         : Boolean
                      [Optional. Default=True]Print intermediate steps
    sp_only         : Boolean
                      [Optional. Default=False] If True, run purely spatial
                      test.
    int_ids         : Boolean
                      [Optional. Default=True] If True, assumes ids in `id_order` are
                      ints.
    Returns
    -------
    quadS

    '''
    write = True
    if not opath:
        opath = shp_link.replace('.shp', '_temp')
        write = False
    # Run the scan
    cas, geo, prm, log = send2satscan(y, shp_link, id_col, opath=opath, \
            sp_win=sp_win, t_win=t_win, stype=stype, rep=rep, weight=weight, \
            tpl_link=tpl_link, verbose=verbose, sp_only=sp_only)
    cls = parse_satscan_output(opath+'_out.txt', p_thres=p_thres)
    if verbose:
        if print_cl_info:
            print(statscan_cluster_info(cls))
    quadS = code_satscan(cls, geo.index.values, cas.time.unique().tolist(),
            p_thres=p_thres, int_ids=int_ids)
    # (Potentially) remove files
    if not write:
        files = ("%s.cas %s.geo %s.prm %s_out.col.shp %s_out.col.shx " \
        "%s_out.col.prj %s_out.col.dbf %s_out.txt")%tuple([opath]*8)
        for f in files.split(' '):
            try:
                _ = check_output(['rm', f])
            except:
                print("Could not remove file: %s"%f)
        if verbose:
            print("SaTScan files removed")
    quadS = code_satscan(cls, geo.index.values, cas.time.unique().tolist(), p_thres=p_thres, int_ids=int_ids)
    return quadS

def send2satscan(y, shp_link, id_col, sp_win=50, t_win=50, stype=1, rep=999,
        opath=None, dates=(0, 23), weight='bag',
        bag_p_poly='../data/bag_per_GRIDCODE.csv',
        tpl_link='satscan_template_normal.prm', verbose=True,
        sp_only=False):
    '''
    Prepare data to ship to SaTScan
    ...

    Arguments
    ---------
    y               : Series
                      Variable MultiIndex'ed with location and time ID,
                      respectively.
    shp_link        : str
                      Path to the shapefile for geography.
    id_col          : str
                      Column in DBF of `shp_link` with IDs to match to `y`.
    sp_win          : float
                      [Optional. Default=50] Maximum percentage of the total
                      dataset to be used for the spatial window of the scan.
    t_win           : float
                      [Optional. Default=50] Maximum percentage of the total
                      dataset to be used for the time window of the scan.
    stype           : int
                      [Optional. Default=1] Type of scan:
                        * 1: high values
                        * 2: low values
    rep             : int
                      [Optional. Default=999] Replications for obtain
                      inference.
    opath           : str
                      [Optional. Default=None] Path to set of files (do not
                      include extension as it will create more than one).
                      If no `opath` is passed, SaTScan is not called and the
                      method returns only the setup. 
    weight          : None/str
                      [Optional. Default='bag'] Switch to use a weighted normal
                      statistic. It may take the following values:

                        * `bag`
                        * `area`
                        * `area_inv`
                        * `bag_dens`
                        * None

    bag_p_poly      : None/str
                      [Optional. Default='../data/bag_per_GRIDCODE.csv']Path to
                      # of BAG buildings per polygon.
    tpl_link        : str
                      [Optional. Default='satscan_template_normal.prm'] Path
                      to prm template
    verbose         : Boolean
                      [Optional. Default=True]Print intermediate steps
    sp_only         : Boolean
                      [Optional. Default=False] If True, run purely spatial
                      test.

    Returns
    -------
    y              : DataFrame
                      Same as passed in
    geo             : DataFrame
                      Table with point coordinates indexed to `id_col`
    prm             : str
                      Content of project file for SaTScan
    '''
    # GEO file
    shp = ps.open(shp_link)
    pts = pd.DataFrame(np.array([poly.centroid for poly in shp]))
    pts.columns = ['X', 'Y']
    pts.index = ps.open(shp_link.replace('.shp', '.dbf')).by_col(id_col)
    pts = pts[['Y', 'X']]
    # CAS file
    cas = y.unstack().reindex(ps.open(shp_link.replace('.shp', '.dbf'))\
            .by_col(id_col))\
            .stack()
    cas = pd.DataFrame({'id': cas.index.get_level_values(0), \
                        'var': cas.values, \
                        'time': cas.index.get_level_values(1)})
    if weight == 'bag':
        pop = pd.read_csv(bag_p_poly, \
                names=['GRIDCODE', 'weight'], index_col=0)
        cas = cas.join(pop, on='id')
    elif weight == 'area':
        areas = pd.DataFrame({'area': [poly.area for poly in ps.open(shp_link)]}, \
                index=ps.open(shp_link.replace('.shp', '.dbf')).by_col(id_col))
        cas = cas.join(areas, on='id')
        cas['area'] = cas['area'] * 100000 #More legible number
    elif weight == 'area_inv':
        areas = pd.DataFrame({'area': [poly.area for poly in ps.open(shp_link)]}, \
                index=ps.open(shp_link.replace('.shp', '.dbf')).by_col(id_col))
        cas = cas.join(areas, on='id')
        cas['area_inv'] = 100000 / cas['area'] #More legible number
    elif weight == 'bag_dens':
        pop = pd.read_csv(bag_p_poly, \
                names=['GRIDCODE', 'weight'], index_col=0)
        cas = cas.join(pop, on='id')
        areas = pd.DataFrame({'area': [poly.area for poly in ps.open(shp_link)]}, \
                index=ps.open(shp_link.replace('.shp', '.dbf')).by_col(id_col))
        cas = cas.join(areas, on='id')
        cas['area'] = cas['area'] * 100000 #More legible number
        cas['weight'] = cas['weight'] / cas['area']
    else:
        if verbose:
            print('No weights specified')
        cas['weight'] = 1.
    cas['one'] = 1
    if sp_only:
        cas['time'] = 0
    cas = cas[['id', 'one', 'time', 'var', 'weight']]
    # PRM file
    time_id = y.index.get_level_values(1)
    beg, end = time_id.min(), time_id.max()
    try:
        a = int(beg) + int(end)
    except:
        beg, end = 0, 1
    temp = open(tpl_link).read()
    temp = temp.replace('XXXsp_winXXX', str(sp_win))\
            .replace('XXXt_winXXX', str(t_win))\
            .replace('XXXt_begXXX', str(beg))\
            .replace('XXXt_endXXX', str(end))\
            .replace('XXXscan_typeXXX', str(stype))\
            .replace('XXXrepXXX', str(int(rep)))
    # Write
    log = 'No process has been run on SaTScan'
    if opath:
        cas_link = opath + '.cas'
        geo_link = opath + '.geo'
        prm_link = opath + '.prm'
        temp = temp.replace('XXXcasXXX', cas_link)\
                .replace('XXXgeoXXX', geo_link)\
                .replace('XXXoutXXX', opath+'_out.txt')
        #---
        _ = pts.to_csv(geo_link, sep=' ', index=True, header=False)
        #---
        cas.to_csv(cas_link, sep=' ', index=False, header=False)
        #---
        fo = open(prm_link, 'w')
        fo.write(temp)
        fo.close()
        if verbose:
            print('\nProcess sent to SaTScan, computing...\n')
        t0 = time.time()
        log = check_output([satscan, prm_link])
        t1 = time.time()
        if verbose:
            print('Job finished:\n\tCAS file in %s\n\tGEO file in %s\n\tPRM file in%s'\
                    %(cas_link, geo_link, prm_link))
            print 'Total time: %.2f minutes\n'%((t1-t0) / 60.)
    return cas, pts, temp, log

def parse_satscan_output(link, p_thres=0.05):
    '''
    Parse the string block of a cluster in SaTScan output
    ...
    
    Arguments
    ---------
    link            : str
                      Path to output file.
    p_thres         : float
                      [Optional. Default=0.05] Threshold for a cluster to be
                      considered significant.

    Returns
    -------
    sig_clusters    : dict
                      Dictionary where key is the item and the value is a
                      string except for the ids in the cluster(list), radious
                      (float), time frame (list of ints) and the P-value
                      (float).
    '''
    def _parse_cluster(cl):
        d = {'order': cl[0].split('.')[0]}
        for line in cl:
            line = line.strip('\n').strip('\r')
            try:
                parts = line.split(':')
                parts[0] = parts[0].strip('.').strip(' ')
                d[parts[0]] = parts[1]
                key = parts[0]
            except:
                d[key] += ' ' + line + ' '
        d['%s.Location IDs included'%d['order']] = [i.strip(' ') for i in \
                d['%s.Location IDs included'%d['order']].split(',')]
        d['P-value'] = float(d['P-value'])
        e = d['Coordinates / radius'].split('/')[0]\
                .strip(' ').strip('(').strip(')').split(',')
        e = [float(i.strip(' ').split(' ')[0]) for i in e]
        d['epicenter'] = e
        # 1degree = 111.23
        #http://www.ncgia.ucsb.edu/education/curricula/giscc/units/u014/tables/table01.html
        d['radious'] = float(d['Coordinates / radius'].split('/')[1]\
                .strip(' ').strip(' km')) / 111.23
        try:
            d['Time frame'] = map(int, d['Time frame'].split(' to '))
        except:
            pass
        return d

    fo = open(link)
    lines = fo.readlines()
    fo.close()
    sig_clusters = []

    cl = []; onCluster=False
    for i in range(len(lines)):
        if lines[i][1:10]=='.Location':
            onCluster = True
        if (len(lines[i])<2 and onCluster==True) or \
                (lines[i][:10]=='__________' and onCluster==True):
            cl = _parse_cluster(cl)
            if cl['P-value'] < p_thres:
                sig_clusters.append(cl)
            onCluster = False
            cl = []
        if onCluster:
            cl.append(lines[i])
    return sig_clusters

def statscan_cluster_info(cls):
    '''
    Create a string with basic info about the output of SaTScan
    ...

    Arguments
    ---------
    cls     : list
              Output from `parse_satscan_output`

    Returns
    -------
    info    : str
              Message with info
    '''
    template = 'SaTScan output info\n...'
    for cl in cls:
        obs = cl[[i for i in cl.keys() if 'Location IDs included' in i][0]]
        try:
            tframe = map(str, cl['Time frame'])
        except:
            tframe = ('N/A', 'N/A')
        cl_txt = ('\nCluster %s info:\n\tN. of observations: %i\n\t'\
                'Time frame: %s to %s')\
                %(cl['order'], len(obs), tframe[0], tframe[1])
        template += cl_txt
    return template

def code_satscan(cls, id_order, t_order, p_thres=0.05, int_ids=True):
    '''
    Turn cluster dictionary from `parse_satscan_output` into a binary table
    (id x t) that switches on for observations in a cluster and off
    otherwise.
    ...

    Arguments
    ---------
    cls     : dict
              Dictionary where key is the item and the value is a
              string except for the ids in the cluster(list), radious
              (float), time frame (list of ints) and the P-value
              (float).
    id_order: list/array
              Ordered sequence of observations.
    t_order : list
              Ordered sequence of time periods.
    int_ids : Boolean
              [Optional. Default=True] If True, assumes ids in `id_order` are
              ints.

    Returns
    -------
    st_o    : DataFrame
              Table indexed to `ids` and columned to `t` that
              takes one if observation i is part of a cluster at time t, zero
              otherwise.
    '''
    tab = pd.DataFrame(np.zeros((len(id_order), len(t_order))), \
            index=id_order, columns=t_order)
    for cl in cls:
        if cl['P-value'] < p_thres:
            if int_ids:
                ids = map(int, \
                        cl[[i for i in cl.keys() if 'Location IDs included' in i][0]])
            else:
                ids = cl[[i for i in cl.keys() if 'Location IDs included' in i][0]]
            try:
                b, e = cl['Time frame']
                for h in t_order[t_order.index(b): t_order.index(e)+1]:
                    tab.loc[ids, h] = 1
            except:
                tab.loc[ids, :] = 1
                pass
    return tab

def check_repeats(cls):
    '''
    Extract polygons that are part of a cluster and count how many are
    repeated
    ...

    Arguments
    ---------
    cls         : list
                  Output from `parse_satscan_output`

    Returns
    -------
    clustered   : list
                  IDs in any cluster (if repeats>1, some are repeated
    repeats     : int
                  N. of cases where a polygon appears more than once in a
                  cluster
    '''
    clustered = []
    repeats = 0
    for i, cl in enumerate(cls):
        for j in cl['%i.Location IDs included'%(i+1)]:
            if j not in clustered:
                clustered.append(j)
            else:
                repeats += 1
    return clustered, repeats

#---

