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
public class NoBinario extends No {
    
    
    public NoBinario(int dado){
        filhos = new No[2];
        this.chave = dado;
    }

    @Override
    public boolean inserirNo(No n) {
        int esqOUdir=0;
        if(n.chave == chave)
            return false;
        
        if(n.chave > chave){
            //inserir a direita
            esqOUdir=1;
        }
        if(filhos[esqOUdir] == null){
            filhos[esqOUdir] = n;
            return true;
        }else{
            return filhos[esqOUdir].inserirNo(n);   
        }
    }

    @Override
    public No buscarNo(int chave) {
        if(chave == this.chave)
            return this;
        
        int esqOUdir=0;
        if(chave > this.chave){
            esqOUdir = 1;
        }
        if(this.filhos[esqOUdir] == null)
            return null;
        else
            return this.filhos[esqOUdir].buscarNo(chave);
        
    }

    
    
    
    @Override
    public void imprimirPreOrdem() {
        System.out.print(chave+" ");
        
        if(filhos[0] != null)
            filhos[0].imprimirPreOrdem();
        
        if(filhos[1] != null)
            filhos[1].imprimirPreOrdem();
    }
    
    
    
    
}
