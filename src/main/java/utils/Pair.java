package utils;

public class Pair<T1, T2> {
    private T1 first;
    private T2 second;

    public Pair(T1 first, T2 second){
        this.first = first;
        this.second = second;
    }

    public T1 key(){
        return first;
    }

    public T2 value(){
        return second;
    }

    public Boolean equals(Pair<T1, T2> other){
        return this.first.equals(other.first) && this.second.equals(other.second);
    }
}
