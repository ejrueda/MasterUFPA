/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package projetosii;

/**
 *
 * @author Gustavo
 */
public class Arvore {
    
    private No raiz;

    public Arvore(No raiz) {
        this.raiz = raiz;
    }
    
    
    
    public void imprimirPreOrdem(){
        raiz.imprimirPreOrdem();
    }
    
    public boolean inserirNo(No no){
        return raiz.inserirNo(no);
    }
    
    public No buscarNo(int chave){
        return raiz.buscarNo(chave);
    }
    
}
