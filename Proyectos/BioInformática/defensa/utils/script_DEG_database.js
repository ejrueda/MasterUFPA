function download_csv(data, colunm_names, file_name) {
    // this function allows you to download a js array in csv format
    var csv = colunm_names
    data.forEach(function(row) {
            csv += row.join(',');
            csv += "\n";
    });
 
    console.log(csv);
    var hiddenElement = document.createElement('a');
    hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv);
    hiddenElement.target = '_blank';
    hiddenElement.download = file_name;
    hiddenElement.click();
}

// link_DEG: link of the organism that you want download
var link_DEG = "http://origin.tubic.org/deg/public/index.php/organism/bacteria/DEG1018.html?lineage=bacteria&id=DEG1018&page=";
// num_subpages = number of subpages in the table of the essential genes
var num_subpages = 11;
var info_genes = [];
for (i=0; i<num_subpages; i++){
    link_subpage = link_DEG + String(i+1);
    //to navigate to the subpage
    window.location.assign(link_subpage);
    // to get the table of genes
    var genes_table = document.getElementsByTagName('table')[1];
    for (g=0; g<genes_table.rows.length-1; g++){
        info_genes[info_genes.length+g] = genes_table.rows[g+1].innerText.trim().replace(/,/g, "*").split("\n");
    }
}
columns = 'No,DEG ID, Gene Name, Function, Organism\n'
download_csv(info_genes, columns, "test.csv");