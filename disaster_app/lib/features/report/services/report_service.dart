import '../models/report_model.dart';

abstract class ReportService {
  Future<void> submitReport(ReportModel report);
}
