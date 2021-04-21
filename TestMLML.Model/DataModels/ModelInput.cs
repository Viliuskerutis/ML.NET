//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using Microsoft.ML.Data;

namespace TestMLML.Model.DataModels
{
    public class ModelInput
    {
        [ColumnName("week_day"), LoadColumn(0)]
        public float Week_day { get; set; }


        [ColumnName("hour"), LoadColumn(1)]
        public float Hour { get; set; }


        [ColumnName("rout_count"), LoadColumn(2)]
        public float Rout_count { get; set; }


        [ColumnName("temperature"), LoadColumn(3)]
        public float Temperature { get; set; }


        [ColumnName("visibility"), LoadColumn(4)]
        public float Visibility { get; set; }

        //[ColumnName("humidity"), LoadColumn(5)]
        //public float Humidity { get; set; }

        [ColumnName("distance"), LoadColumn(5)]
        public float Distance { get; set; }


    }
}
