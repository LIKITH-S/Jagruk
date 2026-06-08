import 'package:flutter/material.dart';

import 'package:flutter_bloc/flutter_bloc.dart';

import 'features/home/screens/home_screen.dart';

import 'features/alerts/bloc/alert_bloc.dart';
import 'features/alerts/bloc/alert_event.dart';
import 'features/alerts/services/mock_alert_service.dart';

import 'features/safety/bloc/safety_bloc.dart';
import 'features/safety/bloc/safety_event.dart';
import 'features/safety/services/local_safety_service.dart';

import 'features/report/bloc/report_bloc.dart';
import 'features/report/services/mock_report_service.dart';

import 'features/news/bloc/news_bloc.dart';
import 'features/news/bloc/news_event.dart';
import 'features/news/services/mock_news_service.dart';

void main() {
  runApp(const DisasterApp());
}

class DisasterApp extends StatelessWidget {
  const DisasterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiBlocProvider(
      providers: [
        BlocProvider(
          create: (_) => AlertBloc(MockAlertService())..add(LoadAlerts()),
        ),

        BlocProvider(
          create: (_) => SafetyBloc(LocalSafetyService())..add(LoadSafety()),
        ),

        BlocProvider(create: (_) => ReportBloc(MockReportService())),

        BlocProvider(
          create: (_) => NewsBloc(MockNewsService())..add(LoadNews()),
        ),
      ],

      child: MaterialApp(
        debugShowCheckedModeBanner: false,

        title: 'Disaster Intelligence',

        theme: ThemeData(primarySwatch: Colors.red),

        home: const HomeScreen(),
      ),
    );
  }
}
