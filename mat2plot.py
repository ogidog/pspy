#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:20:27 2018

@author: babaymc
"""
import os
import sys
import numpy as np
import json
import scipy.io
import folium
from folium import plugins
import branca
from branca.element import Template, MacroElement
# import collections

def folium_legend(folium_map, legend_str, data):
    dmin = np.min(data)
    dmax = np.max(data)
    dhmin = dmin / 2
    dhmax = dmax / 2
    
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
    
    macro = MacroElement()
    macro._template = Template(template)
    
    folium_map.get_root().add_child(macro)
    
def mat2plot(path_to, caption, ftype, ptype, lonlat, ps):
        
    if ftype in ['json', 'html']:
        
        fname = path_to + caption + '.' + ftype
        ll = np.array(lonlat, dtype = float)
        
        if ftype == 'json':
            
            fulldata = np.hstack((lonlat, ps))
            
            # Write json serial file
            s = json.dumps(fulldata.tolist())
            textfile = open(fname, 'w')
            textfile.write(s)
            textfile.close()
            
        if ftype == 'html':
            
            fulldata = []
            
            for i in range(len(ps)):
                fulldata.append([lonlat[i][1], lonlat[i][0], ps[i]]);   
            
            # Define center point of view
            avrx = (np.max(lonlat[:,1]) + np.min(lonlat[:,1])) / 2
            avry = (np.max(lonlat[:,0]) + np.min(lonlat[:,0])) / 2
            
            # Build base map
            mapview = folium.Map(location = [avrx, avry], zoom_start = 12)
             
            if ptype == 'heatmap':
                folium.plugins.HeatMap(np.array(fulldata, dtype = float), name = 'MapRadar', min_opacity = 0.5, blur = 0, radius = 10).add_to(mapview)
            
                # Build castom legend
                lsp = []
                lsp = np.linspace(np.min(ps), np.max(ps), 7).tolist()
                lsp = np.linspace(-40, 40, 10).tolist()
                cm = branca.colormap.LinearColormap(['cyan', 'yellow', 'red'], vmin = 0, vmax = 1).to_step(index = lsp)
                cm.caption = caption
                cm.add_to(mapview)
                
                # folium_legend(mapview, caption, ps)
            
            if ptype == 'marker':
                
                # Set mesh for ps values
                mesh = np.linspace(np.min(ps), np.max(ps), 11).tolist()
                
                # Define colors for markers
                color_map = branca.colormap.LinearColormap(['blue', 'yellow', 'red']).to_step(11)
                colors = [color_map.rgb_hex_str(1 / len(mesh) * i) for i in range(len(mesh) - 1)]
                
                # Set points
                folium_fg = []
                
                for i in range(len(mesh) - 1):

                    truei = (ps > mesh[i]) * (ps <= mesh[i+1])
                    idx = np.arange(len(ps))[truei]
                    
                    if len(idx) > 0:

                        lgd_txt = '<span style="color: {col};">{txt}</span>'
                        nm = str(mesh[i]) + '...' + str(mesh[i+1]) + ' [' + str(len(idx)) + ']'
                        
                        ffg = folium.FeatureGroup(name = lgd_txt.format(txt = nm, col = colors[i]))
                        
                        for k in idx:
                            
                            folium.CircleMarker(location = [ll[k,1], ll[k,0]], radius = 2, weight = 0, color = colors[i], fill_color = colors[i], fill_opacity = 0.75, fill = True).add_to(ffg)
                        
                        folium_fg.append(ffg)
                        
                for ffg in folium_fg:
                    
                    ffg.add_to(mapview)
                    
                folium.LayerControl(collapsed = False).add_to(mapview)

            # Save to html
            mapview.get_root().render()
            mapview.save(fname)
    
    return()