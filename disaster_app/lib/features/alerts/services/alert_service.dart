import '../models/alert_model.dart';

abstract class AlertService {
  Future<List<Alert>> getAlerts();
}
