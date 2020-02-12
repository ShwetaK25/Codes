#href_yext = r"./NIST_Requirements_newKeywords_v2.xlsx#NIST POLICY!C3"
href_nist = r"./NIST.SP.800-171r1-20180220.pdf#page=25"
href_feature = r"./NIST.feature#text=unauthorized-usage-of-organizational-systems"
href_excel = r"./NIST_Requirements_newKeywords_v2.xlsx#NIST POLICY!C9"

str1 = '''
<!DOCTYPE html>
<html>
    <head>
    </head>
    <body>
        <table border="1">
            <tr>
                <td>Policy document</td>
                <td>Excel</td>
                <td>Feature file</td>
            </tr>
            <tr>
               <td><a href = '''+href_nist+'''>doc</td>
               <td><a href = '''+href_feature+'''>Feature file</td>
               <td><a href = '''+href_excel+''' target="_blank">Excel file</td>
               
            </tr>

        </table>
    </body>
</html>
'''
hs = open(r"C:\Shweta\RA\myhtml.html", 'w')
hs.write(str1)
hs.close()
