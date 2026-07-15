package example;

public class OverloadedService {
    public String find(String key) {
        return key;
    }

    public String find(long id) {
        return Long.toString(id);
    }
}
