package example;

public class OrderController {
    private final OrderService orderService;

    public OrderController(OrderService orderService) {
        this.orderService = orderService;
    }

    public void submit(String orderId, boolean canSubmit) {
        orderService.submit(orderId, canSubmit);
    }
}
