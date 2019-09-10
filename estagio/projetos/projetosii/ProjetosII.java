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
public class ProjetosII {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        NoBinario raiz = new NoBinario(3);
        Arvore arvore = new Arvore(raiz);
        
        
        arvore.inserirNo(new NoBinario(2));
        arvore.inserirNo(new NoBinario(5));
        arvore.inserirNo(new NoBinario(10));
        arvore.inserirNo(new NoBinario(7));
        
        arvore.imprimirPreOrdem();
        
        System.out.println();
        No no5 = arvore.buscarNo(5);
        no5.imprimirPreOrdem();
        
        System.out.println();
        arvore.inserirNo(new NoBinario(4));
        no5.imprimirPreOrdem();
        
        
        System.out.println("Nova inserção com mesma chave: "
                +arvore.inserirNo(new NoBinario(4)));
        
    }
    
}
