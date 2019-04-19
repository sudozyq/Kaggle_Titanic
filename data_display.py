# coding:utf-8

import webbrowser

###################################################
#                   数据可视化                      #
###################################################

# 命名生成的html
GEN_HTML = "titanic.html"
# 打开文件，准备写入
f = open(GEN_HTML, 'w')


message = """
<html>
 <head></head>
 <body> 
  <div class="tableauPlaceholder" id="viz1516349898238" style="position: relative">
   <noscript>
    <a href="#"><img alt="An Overview of Titanic Training Dataset " src="https://public.tableau.com/static/images/Ti/Titanic_data_mining/Dashboard1/1_rss.png" style="border: none" /></a>
   </noscript>
   <object class="tableauViz" style="display:none;"><param name="host_url" value="https%3A%2F%2Fpublic.tableau.com%2F"></param> <param name="embed_code_version" value="3"></param> <param name="site_root" value=""></param><param name="name" value="Titanic_data_mining/Dashboard1"></param><param name="tabs" value="no"></param><param name="toolbar" value="yes"></param><param name="static_image" value="https://public.tableau.com/static/images/Ti/Titanic_data_mining/Dashboard1/1.png"></param> <param name="animate_transition" value="yes"></param><param name="display_static_image" value="yes"></param><param name="display_spinner" value="yes"></param><param name="display_overlay" value="yes"></param><param name="display_count" value="yes"></param><param name="filter" value="publish=yes"></param></object>
  </div> 
  <script type="text/javascript">                    var divElement = document.getElementById('viz1516349898238');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
 </body>
</html>
"""

# 写入文件
f.write(message)
# 关闭文件
f.close()

# 运行完自动在网页中显示
webbrowser.open(GEN_HTML, new=1)
