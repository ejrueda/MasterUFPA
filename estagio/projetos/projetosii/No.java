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
public abstract class No {
    
    protected int chave;
    protected No pai;
    protected No[] filhos;
    
    
    public abstract boolean inserirNo(No n);
    
    public abstract No buscarNo(int chave);
    
    public abstract void imprimirPreOrdem();
    
}
