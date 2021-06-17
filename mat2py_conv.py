#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:20:27 2018

@author: babaymc
"""
import numpy as np;
import json;
import scipy.io;
import folium;
from folium import plugins;
import branca;
from branca.element import Template, MacroElement;
import collections;
import os;
import sys;

def folium_legend(folium_map, legend_str, data):
    dmin = np.min(data);
    dmax = np.max(data);
    dhmin = dmin / 2;
    dhmax = dmax / 2;
    
    template = """
    {% macro html(this, kwargs) %}
    
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>MapRadar</title>
      <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
    
      <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
      <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
      
      <script>
      $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });
    
      </script>
    </head>
    <body>
    
     
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.5);
         border-radius:6px; padding: 10px; font-size:18px; right: 20px; bottom: 20px;'>
         
    <div class='legend-title'> """ + legend_str + """</div>
    <div class='legend-scale'>
      <ul class='legend-labels'>
        <li><span style='background:red;opacity:0.75;'></span>""" + str(dmax)[:5] + """</li>
        <li><span style='background:orange;opacity:0.75;'></span>""" + str(dhmax)[:5] + """</li>
        <li><span style='background:yellow;opacity:0.75;'></span>0</li>
        <li><span style='background:palegreen;opacity:0.75;'></span>""" + str(dhmin)[:5] + """</li>
        <li><span style='background:cyan;opacity:0.75;'></span>""" + str(dmin)[:5] + """</li>
    
      </ul>
    </div>
    </div>
     
    </body>
    </html>
    
    <style type='text/css'>
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""
    
    macro = MacroElement();
    macro._template = Template(template);
    
    folium_map.get_root().add_child(macro);
    
def filename2legend(fn):
    
    fn_conv = {
            'ps_plot_w':     'Wrapped phase, rad',
            'radps_plot_v' : 'Mean displacemant velocity, mm/year',
            'ps_plot_v-d' :  'Mean displacemant velocity <br> minus smoothed dem error, mm/year',
            'ps_plot_v-do' : 'Mean displacemant velocity <br> minus smoothed dem error <br> and orbital ramps, mm/year',
            'ps_plot_u' :    'Unwrapped phase, rad' 
            }
    try:
        s = fn_conv[os.path.split(fn)[1].split('.')[0]];
    except:
        s = 'Unknown'; 
    return(s);
    
def mat2py(path_to_lonlat, path_to_ps, ftype, num = 0):
 
    # Convert mat to numpy array
    lonlat = scipy.io.loadmat(path_to_lonlat)['lonlat'];
    ps = scipy.io.loadmat(path_to_ps)['ph_disp'];
    m, n = ps.shape;
    
    if type(ps[0,0]) is np.complex64:
        
        ps_real = np.zeros((m, n), dtype = np.float64);
        
        for i in range(m):
            for j in range(n):
                ps_real[i,j] = np.angle(ps[i,j], deg = False);
                
    else:
        ps_real = ps;
 
    if num > n - 1:
            num = 0;
            
    if ftype in ['json', 'html']:
    
        if ftype == 'json':
            
            fulldata = np.hstack((lonlat, ps_real));
            
            # Write json serial file
            s = json.dumps(fulldata.tolist());
        
        if ftype == 'html':
            
            fulldata = [];
            x = ps_real[:,num];
            mx = np.mean(x);
            sx = 1 * np.std(x);
            
            for i in range(len(ps)):
                if ps_real[i,num] > -sx and ps_real[i,num] < sx:
                    fulldata.append([lonlat[i][1], lonlat[i][0], ps_real[i,num]]);    
            # Define center point of view
            avrx = (np.max(lonlat[:,1]) + np.min(lonlat[:,1])) / 2;
            avry = (np.max(lonlat[:,0]) + np.min(lonlat[:,0])) / 2;
            
            # Define gradient for heatmap
            #color_map = branca.colormap.LinearColormap(['blue', 'yellow', 'red'], vmin = 0, vmax = 1).to_step(7);
            #grd = collections.defaultdict(dict);
            #for i in range(10):
            #    grd[1 / 10 * i] = color_map.rgb_hex_str(1 / 10 * i);
            
            # Build map
            mapview = folium.Map(location = [avrx, avry], zoom_start = 12);
            folium.plugins.HeatMap(fulldata, name = 'MapRadar', min_opacity = 0.5, blur = 0, radius = 10).add_to(mapview);
            
            # Build legend
            #lsp = [];
            #lsp = np.linspace(np.min(ps_real), np.max(ps_real), 7).tolist();
            #cm = branca.colormap.LinearColormap(['cyan', 'yellow', 'red'], vmin = 0, vmax = 1).to_step(index = lsp);
            #cm.caption = filename2legend(os.path.split(path_to_ps)[1]);
            #cm.add_to(mapview);

            folium_legend(mapview, filename2legend(path_to_ps),ps_real[:,num]);
            
            # Send html to string
            s = mapview.get_root().render();
            # mapview.save(s);
            
    else:
        s = 'Files path or return type error';
    
    return(s);
    
if __name__ == '__main__':
        
    # Get command line options
    opts = sys.argv;
    
    path_to_lonlat = opts[1];
    path_to_ps = opts[2];
    ftype = opts[3];
    out_str = mat2py(path_to_lonlat, path_to_ps, ftype);
    # Write file
    path_list = os.path.split(path_to_ps);
    textfile = open(path_list[0] + '/' + path_list[1].split('.')[0] + '.' + ftype, 'w');
    textfile.write(out_str);
    textfile.close();       